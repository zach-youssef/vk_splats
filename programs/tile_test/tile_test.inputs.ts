// tile_test on the web: the program builds (every program does), but its check is native only
// (tile_test.cpp), so a web run fails at once. The web's tile pipeline is checked by T10's web tests.
import type * as zgal from './zgal/index.ts';
import type { MainInputs } from './tile_test.zgal.ts';

function nativeOnly(): never {
  throw new Error('tile_test checks its readbacks natively only (tile_test.cpp); run the native build');
}

export function mainInputs(_context: zgal.Context): MainInputs {
  return { check: nativeOnly, fail: nativeOnly };
}
