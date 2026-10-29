import GaeaPluginKit
import Qwen21Plugin

// The plugin's worker: Qwen21Plugin's definition over GPP (GaeaPluginKit). `--manifest <path>` writes manifest.json.
await GaeaPluginMain.run(qwen21Plugin)
