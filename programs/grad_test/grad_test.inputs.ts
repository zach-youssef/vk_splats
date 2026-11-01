// grad_test on the web (Node): `check` writes the checked frame's readbacks to the directory
// $GRAD_TEST_READBACKS names, and web-check/grad_test_check.cpp runs grad_test's C++ checks
// (grad_check.h, the native program's) on them afterwards, so the web's backward pass is held to
// the same closed-form oracle without a second port of it. test.sh runs both. An extern can't take
// a path option, hence the variable. `sameGradients` compares bit for bit here, as natively. The
// finite differences run in the app, so they need nothing here.
import * as zgal from './zgal/index.ts';
import { Camera, DensifyStat, Projected, ProjectedGradient, Splat, type MainInputs } from './grad_test.zgal.ts';

type Fs = typeof import('node:fs');

function nodeFs(): Fs {
  const fs = globalThis.process?.getBuiltinModule?.('node:fs') as Fs | undefined;
  if (fs === undefined) throw new Error('grad_test writes its readbacks to files, so it runs under Node only');
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
    check(splats, cameras, projected, gradients, parameterGradients, densify, losses, cameraIndex, shDegree, background,
      target, l1, oracle) {
      // As grad_check.h's check: a frame whose readbacks disagree isn't complete yet
      if (gradients.length !== splats.length || projected.length !== splats.length ||
        parameterGradients.length !== splats.length || densify.length !== splats.length || losses.length !== 3) {
        return false;
      }
      const dir = globalThis.process?.env?.GRAD_TEST_READBACKS;
      if (!dir) throw new Error('GRAD_TEST_READBACKS must name the directory for the readbacks (test.sh sets it)');
      const fs = nodeFs();
      fs.mkdirSync(dir, { recursive: true });
      const write = <T>(name: string, t: zgal.Type<T>, elements: readonly T[]) =>
        fs.writeFileSync(`${dir}/${name}.bin`, bytes(t, elements));
      write('splats', Splat, splats);
      write('cameras', Camera, cameras);
      write('projected', Projected, projected);
      write('gradients', ProjectedGradient, gradients);
      write('parameter-gradients', Splat, parameterGradients);
      write('densify', DensifyStat, densify);
      write('losses', zgal.float, losses);
      // The floats as doubles' shortest text, which parses back to the same floats
      fs.writeFileSync(`${dir}/check.txt`,
        `${cameraIndex} ${shDegree} ${background.join(' ')} ${target.join(' ')} ${l1 ? 1 : 0} ${oracle ? 1 : 0}\n`);
      console.log(`grad_test: wrote the checked frame's readbacks to ${dir}`);
      return true;
    },
    sameGradients: (a, b) => same(ProjectedGradient, a, b),
    fail() {
      globalThis.process?.exit(1);
      throw new Error('grad_test failed');
    },
  };
}
