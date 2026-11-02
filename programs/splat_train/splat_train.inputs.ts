// splat_train on the web (Node): the PLY export, as splat_train.cpp's (the same bytes), to
// $SPLAT_TRAIN_PLY (splat_train.ply by default), and the failure exit.
import * as zgal from './zgal/index.ts';
import { type MainInputs, type Splat } from './splat_train.zgal.ts';

type Fs = typeof import('node:fs');

function nodeFs(): Fs {
  const fs = globalThis.process?.getBuiltinModule?.('node:fs') as Fs | undefined;
  if (fs === undefined) throw new Error('splat_train writes its PLY to a file, so it runs under Node only');
  return fs;
}

// The INRIA PLY's header (splat_train.cpp's)
function header(count: number): string {
  const names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2'];
  for (let k = 0; k < 45; k++) names.push(`f_rest_${k}`);
  names.push('opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3');
  return `ply\nformat binary_little_endian 1.0\nelement vertex ${count}\n` +
    names.map((n) => `property float ${n}\n`).join('') + 'end_header\n';
}

const properties = 62;

export function mainInputs(_context: zgal.Context): MainInputs {
  return {
    exportPly(splats: readonly Splat[]) {
      const path = globalThis.process?.env?.SPLAT_TRAIN_PLY || 'splat_train.ply';
      const head = new TextEncoder().encode(header(splats.length));
      const body = new Float32Array(splats.length * properties);
      splats.forEach((s, i) => {
        const p = body.subarray(i * properties, (i + 1) * properties);
        p.set(s.position, 0);
        for (let c = 0; c < 3; c++) p[6 + c] = s.sh[c];
        for (let c = 0; c < 3; c++) for (let k = 1; k < 16; k++) p[9 + 15 * c + k - 1] = s.sh[3 * k + c];
        p[54] = s.opacity;
        p.set(s.logScale, 55);
        p.set(s.rotation, 58);
      });
      const bytes = new Uint8Array(head.length + body.byteLength);
      bytes.set(head, 0);
      bytes.set(new Uint8Array(body.buffer), head.length);  // little-endian on every Node platform
      nodeFs().writeFileSync(path, bytes);
      console.log(`splat_train: wrote ${splats.length} splats to ${path}`);
    },
    fail() {
      globalThis.process?.exit(1);
      throw new Error('splat_train failed');
    },
  };
}
