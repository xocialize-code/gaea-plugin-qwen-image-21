import Foundation
import GaeaPluginKit
import GaeaProtocol
import ImageIO
import MLXQwenImage21
import MLXToolKit

// gaea-plugin-qwen-image-21: Qwen-Image-2.1 (Qwen Research licence: non-commercial, research and evaluation only) as a
// Gaea research plugin, on qwen-image21-swift through MLXServeEngine (Gaea AB-T-0374).
//
// Two checkpoints of one model, as mlx-image-runtime's catalog serves them (AB-D-0116):
// - qwen-image-2.1: the Turbo checkpoint, the 2.1 DiT distilled to its own eight fixed sigmas. A request's steps is
//   ignored and there is no guidance;
// - qwen-image-2.1-base: the original, 40 shifted steps; true CFG only with guidance above 1 and a negative prompt.
// Both are a 7B single-stream block-causal DiT with a Qwen3-VL-8B conditioner and a 64-channel RGBA VAE: text to image,
// and instruction editing from 1–10 references. Output is an RGBA PNG; transparency is asked for in the prompt
// ("This is an RGBA image with transparency. … the background is transparent."), not by a parameter.
//
// The weights are not permissive (LicenseRef-Qwen-Research, AB-D-0085), so Gaea makes every offer named-only and opt-in
// by itself; the offers also say `named_only`. One lane each, bf16, with the package's footprints. Weights come from the
// models root Gaea sends in configure; the engine materializes a missing snapshot.

/// The measured envelope (QwenImage21Envelope): multiples of 32; text to image up to 2400×1792 in area (the largest
/// size on the model card, decoded in tiles above 1024²); an edit up to 1024² from 1–10 references.
public let grid = 32, minSide = 256, maxSide = 2752
public let maxGeneratePixels = QwenImage21Envelope.maxTextToImagePixels
public let maxEditPixels = QwenImage21Envelope.maxTargetPixels
public let maxReferences = QwenImage21Envelope.maxReferenceImages

public enum Model: String, CaseIterable, Sendable {
    case turbo = "qwen-image-2.1"
    case base = "qwen-image-2.1-base"

    public var title: String { self == .turbo ? "Qwen-Image-2.1" : "Qwen-Image-2.1 Base" }

    /// Turbo samples its checkpoint's eight fixed sigmas whatever a request says; the base runs 40 shifted steps.
    public var distilled: Bool { self == .turbo }
    public var defaultSteps: Int { self == .turbo ? 8 : 40 }

    /// Both are sampled without guidance by default (the catalog's 1.0).
    public var defaultGuidance: Double { 1 }

    public var registration: PackageRegistration {
        self == .turbo ? QwenImage21TurboPackage.registration : QwenImage21Package.registration
    }

    /// The lanes: bf16 (DiT bf16, VAE fp32) is the package's only quant.
    public var quants: [Quant] { [.bf16] }

    /// The package configuration for one lane; qwen21-inproc builds the same one. Empty paths: the store resolves
    /// xocialize/Qwen-Image-2.1-Turbo (Turbo's DiT), xocialize/Qwen-Image-2.1 (the base DiT, and the fp32 VAE both use)
    /// and Qwen/Qwen3-VL-8B-Instruct (the conditioner).
    public func configuration(quant: Quant) -> any PackageConfiguration {
        self == .turbo ? QwenImage21TurboConfiguration() : QwenImage21Configuration()
    }
}

/// The MES size a job asks for (`size`, or `aspect` with `megapixels`, Image profile §2), else `fallback`.
func requestedSize(_ job: Job, fallback: (Int, Int)) throws -> (Int, Int) {
    if let size = job.param("size") {
        guard let w = size["width"]?.intValue, let h = size["height"]?.intValue, w > 0, h > 0 else {
            throw GPPError(.invalidRequest, "size is {width, height} in pixels.")
        }
        return (w, h)
    }
    if let aspect = job.param("aspect")?.stringValue {
        let parts = aspect.split(separator: ":").compactMap { Double($0) }
        guard parts.count == 2, parts[0] > 0, parts[1] > 0 else { throw GPPError(.invalidRequest, "aspect is W:H, e.g. 16:9.") }
        let pixels = (job.param("megapixels")?.doubleValue ?? 1.0) * 1_000_000
        let height = (pixels * parts[1] / parts[0]).squareRoot()
        return (Int(height * parts[0] / parts[1]), Int(height))
    }
    return fallback
}

/// An image file's pixel size (PNG or JPEG), nil when it can't be read. An edit without a size takes the first image's.
public func imageSize(_ url: URL) -> (Int, Int)? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let width = properties[kCGImagePropertyPixelWidth] as? Int,
          let height = properties[kCGImagePropertyPixelHeight] as? Int else { return nil }
    return (width, height)
}

/// Fits a size to the envelope: scaled down to the area cap (2400×1792 for text to image, 1024² for an edit), keeping
/// the aspect, then floored to multiples of 32. A change is reported as `size_snapped`.
public func snapped(_ size: (Int, Int), edit: Bool) throws -> (width: Int, height: Int, warning: Warning?) {
    let cap = edit ? maxEditPixels : maxGeneratePixels
    var (w, h) = (Double(size.0), Double(size.1))
    if w * h > Double(cap) {
        let scale = (Double(cap) / (w * h)).squareRoot()
        w *= scale
        h *= scale
    }
    let width = Int(w) / grid * grid, height = Int(h) / grid * grid
    let area = edit ? "1024²" : "2400×1792"
    guard min(width, height) >= minSide, max(width, height) <= maxSide else {
        throw GPPError(.invalidRequest, "Qwen-Image-2.1 draws \(minSide)–\(maxSide) px a side and at most \(area) in area"
            + (edit ? " for an edit" : "") + "; \(size.0)×\(size.1) doesn't fit.", detail: ["param": "size"])
    }
    let warning = (width, height) == size ? nil : Warning(
        code: "size_snapped", message: "Drew \(width)×\(height) for \(size.0)×\(size.1): multiples of 32, at most \(area) in area.",
        param: "size")
    return (width, height, warning)
}

/// How a run samples, resolved the same way by the worker and qwen21-inproc: steps, guidance, the negative prompt, and
/// the warnings for what a model ignores. `steps`, `guidance` and `negative` are what the request asked for, or nil.
public struct Sampling: Sendable {
    public var steps: Int
    public var guidance: Double
    public var negative: String?
    public var warnings: [Warning] = []

    public init(_ model: Model, steps: Int?, guidance: Double?, negative: String?) throws {
        let negative = negative.flatMap { $0.isEmpty ? nil : $0 }
        guard model == .base else {
            // Turbo: the checkpoint's grid is the schedule (the package ignores a request's steps), and no guidance.
            self.steps = model.defaultSteps
            self.guidance = model.defaultGuidance
            self.negative = nil
            if steps != nil {
                warnings.append(Warning(code: "param_ignored",
                                        message: "Qwen-Image-2.1 is the Turbo checkpoint: it samples its own 8 fixed steps. "
                                            + "qwen-image-2.1-base takes x-qwen21.steps.",
                                        param: "x-qwen21.steps"))
            }
            if guidance != nil {
                warnings.append(Warning(code: "param_ignored",
                                        message: "Qwen-Image-2.1 (Turbo) is distilled to run without guidance.",
                                        param: "x-qwen21.guidance"))
            }
            if negative != nil {
                warnings.append(Warning(code: "negative_prompt_ignored",
                                        message: "Qwen-Image-2.1 (Turbo) runs without CFG, so a negative prompt does nothing; "
                                            + "qwen-image-2.1-base honours it with x-qwen21.guidance above 1.",
                                        param: "negative_prompt"))
            }
            return
        }
        self.steps = steps ?? model.defaultSteps
        guard (1...100).contains(self.steps) else { throw GPPError(.invalidRequest, "x-qwen21.steps is 1–100.") }
        let guidance = guidance ?? model.defaultGuidance
        guard (1...20).contains(guidance) else { throw GPPError(.invalidRequest, "x-qwen21.guidance is 1–20.") }
        // The package runs true CFG (two DiT forwards a step) only when both are there; anything else samples without.
        switch (guidance > 1, negative) {
        case (true, .some):
            self.guidance = guidance
            self.negative = negative
        case (true, nil):
            self.guidance = 1
            warnings.append(Warning(code: "param_ignored",
                                    message: "True CFG needs a negative prompt; without one Qwen-Image-2.1 Base ran without guidance.",
                                    param: "x-qwen21.guidance"))
        case (false, .some):
            self.guidance = 1
            warnings.append(Warning(code: "negative_prompt_ignored",
                                    message: "A negative prompt takes effect only with x-qwen21.guidance above 1 (true CFG).",
                                    param: "negative_prompt"))
        case (false, nil):
            self.guidance = 1
        }
    }
}

/// The text-to-image request for one image; the worker sends image i with `seed + i`, and qwen21-inproc sends this too.
public func t2iRequest(prompt: String, width: Int, height: Int, sampling: Sampling, seed: UInt64) -> T2IRequest {
    T2IRequest(prompt: prompt, negativePrompt: sampling.negative, width: width, height: height, steps: sampling.steps,
               guidanceScale: sampling.guidance, seed: seed)
}

/// The edit request: the references in instruction order ("<image1>", "<image2>"…), the output at an explicit size.
public func editRequest(images: [Image], instruction: String, width: Int, height: Int, sampling: Sampling,
                        seed: UInt64) -> IEditRequest {
    IEditRequest(images: images, prompt: instruction, negativePrompt: sampling.negative, width: width, height: height,
                 steps: sampling.steps, guidanceScale: sampling.guidance, seed: seed)
}

/// The run's settings shared by generate and edit: sampling, seed and count, with their warnings.
struct Shared {
    var sampling: Sampling
    var seed: UInt64
    var count: Int
    var warnings: [Warning]

    /// `counted`: image.generate takes MES `count`; image.edit has one output and no count.
    init(_ job: Job, model: Model, counted: Bool = true) throws {
        sampling = try Sampling(model, steps: job.param("x-qwen21.steps")?.intValue,
                                guidance: job.param("x-qwen21.guidance")?.doubleValue,
                                negative: job.param("negative_prompt")?.stringValue)
        warnings = sampling.warnings
        count = counted ? job.param("count")?.intValue ?? 1 : 1
        guard (1...8).contains(count) else { throw GPPError(.invalidRequest, "count is 1–8.") }
        if job.param("prompt_enhance")?.stringValue == "on" {
            warnings.append(Warning(code: "prompt_enhance_ignored", message: "This plugin doesn't rewrite prompts; Gaea's enhancer does.",
                                    param: "prompt_enhance"))
        }
        // Image profile §2: a request without a seed still gets one, reported in effective.
        seed = job.seed ?? UInt64.random(in: 0..<(1 << 31))
    }

    func effective(width: Int, height: Int, model: Model, counted: Bool = true) -> [String: JSONValue] {
        var values: [String: JSONValue] = [
            "size": ["width": .int(Int64(width)), "height": .int(Int64(height))], "seed": .int(Int64(seed)),
        ]
        if counted { values["count"] = .int(Int64(count)) }
        if model == .base { values["x-qwen21.steps"] = .int(Int64(sampling.steps)) }
        if sampling.guidance > 1 { values["x-qwen21.guidance"] = .double(sampling.guidance) }
        if let negative = sampling.negative { values["negative_prompt"] = .string(negative) }
        return values
    }
}

/// MES image.generate: one engine call per image, each with seed + i.
func generate(_ model: Model) -> EngineOperation {
    EngineOperation("image.generate", outputNames: ["image": "images"]) { job in
        guard let prompt = try job.text("prompt"), !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw GPPError(.invalidRequest, "image.generate needs a prompt.")
        }
        let size = try snapped(try requestedSize(job, fallback: (1024, 1024)), edit: false)
        var shared = try Shared(job, model: model)
        if let warning = size.warning { shared.warnings.insert(warning, at: 0) }
        let effective = shared.effective(width: size.width, height: size.height, model: model)
        return (0..<shared.count).map { i in
            EngineCall(t2iRequest(prompt: prompt, width: size.width, height: size.height, sampling: shared.sampling,
                                  seed: shared.seed + UInt64(i)),
                       effective: effective, warnings: shared.warnings)
        }
    }
}

/// MES image.edit: 1–10 references in instruction order ("<image1>", "<image2>"… or "image 1"), drawn at the first
/// image's size fitted to 1024² (Image profile §3). One output. The package resizes each reference to 1024² in area.
func edit(_ model: Model) -> EngineOperation {
    EngineOperation("image.edit", outputNames: ["image": "image"]) { job in
        guard let instruction = try job.text("instruction") ?? job.text("prompt"), !instruction.isEmpty else {
            throw GPPError(.invalidRequest, "image.edit needs an instruction.")
        }
        let name = job.inputs["images"] == nil ? "image" : "images"
        let references = job.inputs[name] ?? []
        guard (1...maxReferences).contains(references.count) else {
            throw GPPError(.invalidRequest, "image.edit takes 1–\(maxReferences) images; got \(references.count).",
                           detail: ["param": "images"])
        }
        guard let first = imageSize(references[0]) else {
            throw GPPError(.invalidRequest, "image.edit needs readable PNG or JPEG images.")
        }
        let size = try snapped(try requestedSize(job, fallback: first), edit: true)
        var shared = try Shared(job, model: model, counted: false)
        if let warning = size.warning { shared.warnings.insert(warning, at: 0) }
        if job.param("strength") != nil {
            shared.warnings.append(Warning(code: "param_ignored",
                                           message: "Qwen-Image-2.1 edits by reference conditioning from noise, so strength does nothing.",
                                           param: "strength"))
        }
        let images = try references.indices.map { try job.image(name, index: $0) }
        return [EngineCall(editRequest(images: images, instruction: instruction, width: size.width, height: size.height,
                                       sampling: shared.sampling, seed: shared.seed),
                           effective: shared.effective(width: size.width, height: size.height, model: model, counted: false),
                           warnings: shared.warnings)]
    }
}

public func lanes(_ model: Model) -> [LaneDefinition] {
    model.quants.map { quant in
        LaneDefinition(id: quant.rawValue, default: quant == .bf16, configuration: model.configuration(quant: quant))
    }
}

/// The weights' licence (AB-D-0085): research and evaluation only (§1(i), §2), its attribution in the notice (§3, §4b).
let qwenResearch: JSONValue = [
    "spdx": "LicenseRef-Qwen-Research", "commercial_use": false,
    "terms_url": "https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE",
    "notice": "Qwen Research License: non-commercial research and evaluation use only. Built with Qwen.",
]
let limits = PluginManifest.Limits(maxPixels: maxGeneratePixels, minSide: minSide, maxSide: maxSide, sizeMultiple: grid)
let editLimits = PluginManifest.Limits(maxPixels: maxEditPixels, minSide: minSide, maxSide: maxSide, sizeMultiple: grid,
                                       maxImages: maxReferences)

/// The MES extensions: the base's steps and guidance. Turbo has none; Gaea passes a worker only the ones its offer names.
func params(_ model: Model) -> JSONValue {
    var properties: [String: JSONValue] = [:]
    if model == .base {
        properties["x-qwen21.steps"] = [
            "type": "integer", "minimum": 1, "maximum": 100, "default": .int(Int64(model.defaultSteps)),
            "description": "Denoising steps; more is slower and finer.",
            "x-mes-ui": ["widget": "slider", "label": "Steps", "advanced": true],
        ]
        properties["x-qwen21.guidance"] = [
            "type": "number", "minimum": 1, "maximum": 20, "default": .double(model.defaultGuidance),
            "description": .string("True classifier-free guidance, against the negative prompt. 1 (the default) turns it "
                + "off; above 1 it needs a negative prompt and runs two passes a step."),
            "x-mes-ui": ["widget": "slider", "label": "Guidance", "advanced": true],
        ]
    }
    return ["type": "object", "properties": .object(properties), "additionalProperties": false]
}

func offerModel(_ model: Model, edit: Bool) -> JSONValue {
    let notes: String = switch (model, edit) {
    case (.turbo, false): "The Turbo checkpoint: 8 fixed steps without guidance, so steps and negative prompts do nothing. "
        + "RGBA PNG; ask for transparency in the prompt. Multiples of 32, up to 2400×1792 in area."
    case (.base, false): "The original 40-step checkpoint; true CFG with x-qwen21.guidance above 1 and a negative prompt. "
        + "About 5× Turbo's time: for layout-critical typography. RGBA PNG. Multiples of 32, up to 2400×1792 in area."
    case (.turbo, true): "Instruction editing in 8 fixed steps from 1–10 images, named <image1>, <image2>… in the "
        + "instruction. Drawn from noise at the first image's size fitted to 1024²; strength does nothing."
    case (.base, true): "Instruction editing in 40 steps from 1–10 images, named <image1>, <image2>… in the instruction; "
        + "true CFG with guidance above 1 and a negative prompt. At the first image's size fitted to 1024²."
    }
    return [
        "id": .string(model.rawValue), "title": .string(model.title), "license": qwenResearch,
        "memory_class": "xlarge", "speed_class": .string(model.distilled ? "standard" : "slow"),
        "quality_tiers": model.distilled ? ["standard"] : ["best"], "named_only": true,
        "notes": .string("Research plugin, named-only and opt-in (Qwen Research licence: non-commercial). " + notes),
    ]
}

public let qwen21Plugin = GaeaPluginDefinition(
    id: "com.xocialize.gaea.qwen-image-21",
    version: "0.1.0",
    minGaea: "0.2.0",
    team: "CFZS6RXUCP",
    repo: "https://github.com/xocialize/gaea-plugin-qwen-image-21",
    title: "Qwen-Image-2.1",
    summary: "Qwen-Image-2.1 (Qwen Research licence: non-commercial, research and evaluation only): a 7B block-causal DiT "
        + "with Qwen3-VL-8B conditioning and an RGBA VAE. The 8-step Turbo and the 40-step base, for generation with "
        + "native transparency and editing from up to 10 references. A research plugin: named-only and opt-in.",
    license: "Apache-2.0",
    executable: "gaea-qwen-image-21-worker",
    models: [
        .engine(QwenImage21TurboPackage.self, id: Model.turbo.rawValue, title: Model.turbo.title, lanes: lanes(.turbo),
                operations: [generate(.turbo), edit(.turbo)]),
        .engine(QwenImage21Package.self, id: Model.base.rawValue, title: Model.base.title, lanes: lanes(.base),
                operations: [generate(.base), edit(.base)]),
    ],
    offers: Model.allCases.flatMap { model in
        [
            PluginManifest.Offer(operation: "image.generate", profile: "image", model: offerModel(model, edit: false),
                                 params: params(model), defaults: ["count": 1], limits: limits),
            PluginManifest.Offer(operation: "image.edit", profile: "image", model: offerModel(model, edit: true),
                                 params: params(model), limits: editLimits),
        ]
    })
