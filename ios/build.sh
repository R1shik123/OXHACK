#!/bin/bash
# Generate the Xcode project and build/install Sightline on a connected iPhone.
#   DEVELOPMENT_TEAM=ABCDE12345 ./ios/build.sh            # build for device
#   DEVELOPMENT_TEAM=ABCDE12345 ./ios/build.sh install    # build + install on the plugged-in iPhone
#   ./ios/build.sh sim                                    # compile-only check for the simulator (no signing)
set -euo pipefail
cd "$(dirname "$0")"

XCODEGEN=$(command -v xcodegen || echo ../tools/xcodegen/bin/xcodegen)
[[ -x "$XCODEGEN" ]] || { echo "xcodegen missing: see README (brew install xcodegen, or the release zip into tools/)"; exit 1; }
xcodebuild -version >/dev/null 2>&1 || { echo "Xcode missing or not selected: sudo xcode-select -s /Applications/Xcode.app"; exit 1; }

"$XCODEGEN" generate --quiet

if [[ "${1:-}" == "sim" ]]; then
  xcodebuild -project Sightline.xcodeproj -scheme Sightline -configuration Debug \
    -destination 'generic/platform=iOS Simulator' -derivedDataPath build CODE_SIGNING_ALLOWED=NO build | tail -n 20
  exit 0
fi

: "${DEVELOPMENT_TEAM:?Set DEVELOPMENT_TEAM to your Apple Team ID}"
xcodebuild -project Sightline.xcodeproj -scheme Sightline -configuration Debug \
  -destination 'generic/platform=iOS' -derivedDataPath build -allowProvisioningUpdates \
  DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM" build | tail -n 20

if [[ "${1:-}" == "install" ]]; then
  APP=build/Build/Products/Debug-iphoneos/Sightline.app
  DEVICE=$(xcrun devicectl list devices 2>/dev/null | awk '/iPhone/ && /connected|available/ {for (i=1;i<=NF;i++) if ($i ~ /^[0-9A-F-]{25,}$/) print $i}' | head -n1)
  [[ -n "$DEVICE" ]] || { echo "No iPhone found: plug it in, unlock it, trust this Mac, enable Developer Mode"; exit 1; }
  xcrun devicectl device install app --device "$DEVICE" "$APP"
  xcrun devicectl device process launch --device "$DEVICE" "$(grep -m1 PRODUCT_BUNDLE_IDENTIFIER project.yml | sed -E 's/^[^:]*: *([^ #]+).*/\1/')" || true
fi
