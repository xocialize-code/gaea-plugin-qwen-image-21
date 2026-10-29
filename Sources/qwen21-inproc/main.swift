import Foundation
import MLXQwenImage21
import MLXServeCore
import MLXToolKit
import Qwen21Plugin

// The in-process reference for the plugin: one Qwen-Image-2.1 lane loaded by MLXServeEngine in this process, with the
// lane configuration the worker uses (Qwen21Plugin) and the request it would build: the same size snapping, sampling
// and request functions. A cold run then a warm one; the first image is written to --out, and one JSON line of
// timings is printed. Not part of the plugin bundle.
//
//   qwen21-inproc [--model qwen-image-2.1 | qwen-image-2.1-base] [--lane bf16] [--prompt <text>] [--negative <text>]
//                 [--width 1024] [--height 1024] [--steps N] [--guidance G] [--seed 42] [--image <png> ...]
//                 [--models-root <dir>] [--out <png>] [--runs 2]
//
// Each option stands for the MES param the worker reads: --width/--height for `size`, --steps for x-qwen21.steps,
// --guidance for x-qwen21.guidance, --negative for negative_prompt. Leave one out to take the worker's default. An
// edit takes --image (repeatable, 1–10) with the prompt as the instruction; without --width/--height it is drawn at
// the first image's size, fitted to 1024².

var args = Array(CommandLine.arguments.dropFirst())
func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    defer { args.removeSubrange(i...(i + 1)) }
    return args[i + 1]
}
func options(_ name: String) -> [String] {
    var values: [String] = []
    while let value = option(name) { values.append(value) }
    return values
}
func seconds(since start: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - start
    return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
}
func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

let started = ContinuousClock.now
guard let model = Model(rawValue: option("--model") ?? Model.turbo.rawValue) else {
    fail("unknown --model; one of \(Model.allCases.map(\.rawValue))")
}
guard let quant = Quant(rawValue: option("--lane") ?? Quant.bf16.rawValue), model.quants.contains(quant) else {
    fail("unknown --lane; one of \(model.quants.map(\.rawValue))")
}
let prompt = option("--prompt") ?? "a glass teapot on a wooden table, studio light"
let negative = option("--negative")
let width = option("--width").flatMap { Int($0) }
let height = option("--height").flatMap { Int($0) }
let steps = option("--steps").flatMap { Int($0) }
let guidance = option("--guidance").flatMap { Double($0) }
let seed = UInt64(option("--seed") ?? "") ?? 42
let imagePaths = options("--image")
let outPath = option("--out")
let runs = Int(option("--runs") ?? "") ?? 2
let modelsRoot = URL(fileURLWithPath: option("--models-root")
                     ?? NSHomeDirectory() + "/Library/Application Support/MLXEngine/Models", isDirectory: true)
let isEdit = !imagePaths.isEmpty

// The worker's size and sampling (Qwen21Plugin's generate and edit): the first image's size when an edit names none.
let fallback = isEdit ? imageSize(URL(fileURLWithPath: imagePaths[0])) : (1024, 1024)
guard let fallback else { fail("--image \(imagePaths[0]) isn't a readable PNG or JPEG") }
let size = try snapped((width ?? fallback.0, height ?? fallback.1), edit: isEdit)
let sampling = try Sampling(model, steps: steps, guidance: guidance, negative: negative)

var result: [String: Any] = ["model": model.rawValue, "lane": quant.rawValue, "width": size.width, "height": size.height,
                             "steps": sampling.steps, "guidance": sampling.guidance, "seed": seed,
                             "references": imagePaths.count, "warnings": sampling.warnings.map(\.code)]

// The budget Gaea sends a worker on a large Mac (wiresWeights off, the automatic pool cap).
let engine = MLXServeEngine(workerBudget: WorkerBudget(ceilingBytes: 64_000_000_000, wiresWeights: false))
await engine.useModelStore(ModelStore(root: modelsRoot))
let id = PackageID(model.rawValue)
_ = try await engine.register(model.registration, configuration: model.configuration(quant: quant), id: id)

var clock = ContinuousClock.now
_ = try await engine.prepare(isEdit ? .imageEdit : .textToImage, package: id)
result["prepare_s"] = seconds(since: clock)

let images = try imagePaths.map { path in
    Image(format: path.lowercased().hasSuffix(".png") ? .png : .jpeg, data: try Data(contentsOf: URL(fileURLWithPath: path)))
}
// The worker's request, built by the same functions (image 1 of a generate: seed + 0).
let request: any CapabilityRequest = isEdit
    ? editRequest(images: images, instruction: prompt, width: size.width, height: size.height, sampling: sampling, seed: seed)
    : t2iRequest(prompt: prompt, width: size.width, height: size.height, sampling: sampling, seed: seed)

for run in 1...max(1, runs) {
    clock = ContinuousClock.now
    let response = try await engine.run(request, package: id)
    result["run\(run)_s"] = seconds(since: clock)
    let image = (response as? T2IResponse)?.image ?? (response as? IEditResponse)?.image
    if run == 1, let image, let outPath { try image.data.write(to: URL(fileURLWithPath: outPath)) }
}
result["cold_s"] = seconds(since: started) - (runs > 1 ? result["run\(runs)_s"] as! Double : 0)

let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
print(String(decoding: data, as: UTF8.self))
