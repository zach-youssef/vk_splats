#!/bin/sh
# grad_test on the web, under Node: test-web.sh <main.js> <output dir> <web-check dir>. test.sh
# runs it, as it runs the native binary, and checks its readbacks with <web-check dir>/grad_test_check.
exec "$(dirname "$0")/test.sh" "$@"
