#!/bin/sh
# splat_view on the web (Node): the same test as native, test.sh with the web build
exec "$(dirname "$0")/test.sh" "$@"
