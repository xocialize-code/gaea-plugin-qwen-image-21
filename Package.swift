// swift-tools-version: 6.2
import PackageDescription

// gaea-plugin-qwen-image-21: Qwen-Image-2.1 (Qwen Research licence: non-commercial, research and evaluation only) as a
// Gaea research plugin. One worker executable built on GaeaPluginKit over qwen-image21-swift, packaged as a signed
// .gaeaplugin by scripts/build-plugin.sh.
//
// qwen21-inproc is not part of the plugin: it runs the same packages in-process through MLXServeEngine, with the same
// dependency resolution, as the reference a run through Gaea is compared against.
//
// GAEA_PLUGIN_KIT_PATH points the kit at a local checkout while you work on both.
let kitPath = Context.environment["GAEA_PLUGIN_KIT_PATH"]

let package = Package(
    name: "gaea-plugin-qwen-image-21",
    platforms: [.macOS(.v26)], // mlx-engine-swift's MLXToolKit requires macOS 26
    dependencies: [
        kitPath.map { .package(path: $0) }
            ?? .package(url: "https://github.com/xocialize-code/gaea-plugin-kit.git", from: "0.3.0"),
        .package(url: "https://github.com/xocialize/qwen-image21-swift.git", exact: "0.2.0"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift.git", from: "0.72.0"),
    ],
    targets: [
        .target(
            name: "Qwen21Plugin",
            dependencies: [
                .product(name: "GaeaPluginKit", package: "gaea-plugin-kit"),
                .product(name: "MLXQwenImage21", package: "qwen-image21-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ]
        ),
        .executableTarget(
            name: "gaea-qwen-image-21-worker",
            dependencies: [
                "Qwen21Plugin",
                .product(name: "GaeaPluginKit", package: "gaea-plugin-kit"),
            ]
        ),
        .executableTarget(
            name: "qwen21-inproc",
            dependencies: [
                "Qwen21Plugin",
                .product(name: "MLXQwenImage21", package: "qwen-image21-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
