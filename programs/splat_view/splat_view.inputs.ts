// splat_view on the web: the scene's opener, the failure exit (every other check is in the app),
// and --inspect's accum written to $SPLAT_VIEW_ACCUM for the closed-form oracle
// (web-check/view_oracle_check.cpp), under Node only.
//
// The opener (the graph's stream input) streams a stream file through the context's assets
// (context.loadAssetStream: fetched in a page, read from the file under Node), waits for its
// header's 432 bytes and checks it with lib/splat_stream.zgal's checkStreamHeader (the TypeScript
// export of the same function the native opener and the GPU's PlanDecode run), then hands the
// graph a stream replaying those bytes before the rest. Like the native opener (splat_view.cpp) it
// refuses a header checkStreamHeader refuses, or a stream ending within its header, before the
// graph sizes its buffers from the header: the opener's rejection fails the graph's build, so the
// program fails before its first frame. Unlike the native one it reads no PLY (docs/web.md).
import * as zgal from './zgal/index.ts';
import { checkStreamHeader, StreamHeader, streamHeaderSize, streamOk, type MainInputs } from './splat_view.zgal.ts';

type Fs = typeof import('node:fs');

// The stream at path, its header checked; rejects (cancelling the stream) when it's refused
async function openChecked(path: string, context: zgal.Context): Promise<zgal.Stream> {
  const { head, stream } = await zgal.peekStream(await context.loadAssetStream(path), streamHeaderSize);
  const refuse = (message: string): never => {
    stream.cancel();
    throw new Error(message);
  };
  if (head.length < streamHeaderSize) {
    refuse(`it ends within its ${streamHeaderSize}-byte stream header (${head.length} bytes)`);
  }
  const code = checkStreamHeader(zgal.decode(StreamHeader, head, 1)[0]);
  if (code !== streamOk) {
    refuse(`its stream header is one checkStreamHeader refuses, code ${code} (lib/splat_stream.zgal)`);
  }
  return stream;
}

export function mainInputs(_context: zgal.Context): MainInputs {
  return {
    open: openChecked,
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
