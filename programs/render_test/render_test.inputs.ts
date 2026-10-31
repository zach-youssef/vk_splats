// render_test on the web (Node): `check` writes the checked frame's readbacks to the directory
// $RENDER_TEST_READBACKS names, and web-check/render_test_check.cpp runs render_test's C++ checks
// (render_check.h, the native program's) on them afterwards, so the web's rasterizer is held to the
// same CPU blend and closed-form oracle without a second port of them. test.sh runs both. An extern
// can't take a path option (ZGAL has no path or string extern parameters), hence the variable.
// `sameColours` and `sameEntries` compare bit for bit here, as natively.
import * as zgal from './zgal/index.ts';
import { Camera, Projected, Splat, countCapacity, countSize, type MainInputs } from './render_test.zgal.ts';

type Fs = typeof import('node:fs');

function nodeFs(): Fs {
  const fs = globalThis.process?.getBuiltinModule?.('node:fs') as Fs | undefined;
  if (fs === undefined) throw new Error('render_test writes its readbacks to files, so it runs under Node only');
  return fs;
}

// The elements' std430 bytes, without encode's rounding up to 4 bytes
function bytes<T>(t: zgal.Type<T>, elements: readonly T[]): Uint8Array {
  return zgal.encode(t, elements).subarray(0, zgal.stride(t) * elements.length);
}

function same<T>(t: zgal.Type<T>, a: readonly T[], b: readonly T[]): boolean {
  const x = bytes(t, a);
  const y = bytes(t, b);
  return x.length === y.length && x.every((v, i) => v === y[i]);
}

export function mainInputs(_context: zgal.Context): MainInputs {
  return {
    check(splats, cameras, counts, projected, entrySplats, ranges, colours, lastEntries, cameraIndex, background, oracle) {
      // As render_check.h's check: a frame whose readbacks disagree isn't complete yet
      if (counts.length !== countSize || entrySplats.length !== counts[countCapacity]) return false;
      const dir = globalThis.process?.env?.RENDER_TEST_READBACKS;
      if (!dir) throw new Error('RENDER_TEST_READBACKS must name the directory for the readbacks (test.sh sets it)');
      const fs = nodeFs();
      fs.mkdirSync(dir, { recursive: true });
      const write = <T>(name: string, t: zgal.Type<T>, elements: readonly T[]) =>
        fs.writeFileSync(`${dir}/${name}.bin`, bytes(t, elements));
      write('splats', Splat, splats);
      write('cameras', Camera, cameras);
      write('counts', zgal.uint, counts);
      write('projected', Projected, projected);
      write('entrySplats', zgal.uint, entrySplats);
      write('ranges', zgal.uint, ranges);
      write('colours', zgal.vec4, colours);
      write('lastEntries', zgal.uint, lastEntries);
      // The background's floats as doubles' shortest text, which parses back to the same floats
      fs.writeFileSync(`${dir}/check.txt`, `${cameraIndex} ${background.join(' ')} ${oracle ? 1 : 0}\n`);
      console.log(`render_test: wrote the checked frame's readbacks to ${dir}`);
      return true;
    },
    sameColours: (a, b) => same(zgal.vec4, a, b),
    sameEntries: (a, b) => same(zgal.uint, a, b),
    fail() {
      globalThis.process?.exit(1);
      throw new Error('render_test failed');
    },
  };
}
