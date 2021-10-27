import path from 'path';

export function replaceExtension(fullPath: string, ext: string) {
  return path.format({
    ...path.parse(fullPath),
    base: undefined,
    ext: ext,
  });
}
