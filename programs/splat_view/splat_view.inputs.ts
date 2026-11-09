// splat_view on the web: the failure exit (every check is in the app), and --inspect's accum
// written to $SPLAT_VIEW_ACCUM for the closed-form oracle (web-check/view_oracle_check.cpp), under
// Node only
import * as zgal from './zgal/index.ts';
import { type MainInputs } from './splat_view.zgal.ts';

type Fs = typeof import('node:fs');

export function mainInputs(_context: zgal.Context): MainInputs {
  return {
    fail() {
      globalThis.process?.exit(1);
      throw new Error('splat_view failed');
    },
    saveAccum(accum) {
      const path = globalThis.process?.env?.SPLAT_VIEW_ACCUM;
      if (!path) return;
      const fs = globalThis.process?.getBuiltinModule?.('node:fs') as Fs | undefined;
      if (fs === undefined) throw new Error('SPLAT_VIEW_ACCUM is set, but writing a file needs Node');
      // The elements' std430 bytes (8 a pixel), as the native program writes them
      fs.writeFileSync(path, zgal.encode(zgal.uvec2, accum).subarray(0, zgal.stride(zgal.uvec2) * accum.length));
      console.log(`splat_view: wrote accum to ${path}`);
    },
  };
}
