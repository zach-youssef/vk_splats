#!/bin/sh
# splat_train on the web, under Node: test-web.sh <main.js> <output dir>. test.sh runs it as it
# runs the native binary, rendering the ground truth with splat_render's web build.
exec "$(dirname "$0")/test.sh" "$1" "$2"
