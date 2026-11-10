// splat_view on the web: the scene's opener, the failure exit (every other check is in the app),
// and --inspect's accum written to $SPLAT_VIEW_ACCUM for the closed-form oracle
// (web-check/view_oracle_check.cpp), under Node only. The opener streams a stream file through the
// context's assets as it is: unlike the native one (splat_view.cpp) it reads no PLY and doesn't
// check the header first.
import * as zgal from './zgal/index.ts';
import { type MainInputs } from './splat_view.zgal.ts';

type Fs = typeof import('node:fs');

export function mainInputs(_context: zgal.Context): MainInputs {
  return {
    open: (path, context) => context.loadAssetStream(path),
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
