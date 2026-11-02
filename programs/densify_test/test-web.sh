#!/bin/sh
# densify_test on the web, under Node: test-web.sh <main.js> <output dir>. test.sh runs it as it
# runs the native binary (every check is in the app, so the same on both).
exec "$(dirname "$0")/test.sh" "$1" "$2"
