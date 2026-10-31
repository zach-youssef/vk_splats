#!/bin/sh
# splat_render on the web, under Node: test-web.sh <main.js> <output dir>. test.sh runs it as it
# runs the native binary, against the same goldens and PSNR.
exec "$(dirname "$0")/test.sh" "$1" "$2"
