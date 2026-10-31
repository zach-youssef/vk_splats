// render_test on the web: the program builds (every program does), but its checks are native only
// (render_test.cpp), so a web run fails at once. The web's rasterizer is checked by T10's web tests.
import type * as zgal from './zgal/index.ts';
import type { MainInputs } from './render_test.zgal.ts';

function nativeOnly(): never {
  throw new Error('render_test checks its readbacks natively only (render_test.cpp); run the native build');
}

export function mainInputs(_context: zgal.Context): MainInputs {
  return { check: nativeOnly, sameColours: nativeOnly, sameEntries: nativeOnly, fail: nativeOnly };
}
