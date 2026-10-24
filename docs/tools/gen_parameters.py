#!/usr/bin/env python3
"""Generates docs/parameters.md from the comments in src/Parameters.h.

The descriptions and sections come from the header, the default values from the
installed MultiNEAT module (so they are always the real defaults).

usage: python docs/tools/gen_parameters.py
"""
import os
import re

import MultiNEAT as NEAT

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEADER = os.path.join(ROOT, 'src', 'Parameters.h')
OUT = os.path.join(ROOT, 'docs', 'parameters.md')

# C++ member name -> Python attribute name, where they differ
RENAMED = {'pyCustomConstraints': 'CustomConstraints', 'pyBehaviorGetter': 'BehaviorGetter'}
SKIP = {'CustomConstraints'}  # the C++ function pointer
DESCRIPTIONS = {
    'CustomConstraints': 'A Python callable `f(genome) -> bool` that returns True if the genome FAILS '
                         'custom topology constraints. Such genomes are not added to the population.',
    'BehaviorGetter': 'A Python callable `f(genome)` returning a behavior object with a `distance_to(other)` '
                      'method. When set, the compatibility distance between genomes is the distance between '
                      'their behaviors (behavior based speciation).',
}

INTRO = """# Parameters reference

All settings of the algorithm live in a `Parameters` object:

```python
params = NEAT.Parameters()
params.PopulationSize = 150
params.Save('params.txt')   # plain text, one "Name value" per line
params.Load('params.txt')   # returns 0 on success
```

The parameters are copied into the `Population` when it is created; later changes go
through `pop.Parameters`. Trait parameters are set with `SetNeuronTraitParameters`,
`SetLinkTraitParameters` and `SetGenomeTraitParameters` (see [Traits](traits.md)).

Probabilities are in the range 0..1. *(This page is generated from `src/Parameters.h` by
`docs/tools/gen_parameters.py`.)*
"""


def fmt(v):
    if isinstance(v, bool):
        return 'True' if v else 'False'
    if isinstance(v, float):
        return ('%.6g' % v)
    if v is None:
        return 'None'
    return str(v)


def parse():
    text = open(HEADER, encoding='utf8').read()
    body = text[text.index('class Parameters'):text.index('    Parameters();')]
    lines = body.splitlines()

    sections = []
    current = None
    comment = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # a section header is a comment line surrounded by //// lines
        if re.match(r'^/{6,}$', line) and i + 2 < len(lines) and re.match(r'^/{6,}$', lines[i + 2].strip()):
            title = lines[i + 1].strip().strip('/').strip()
            if title and title != 'Members':
                current = (title, [])
                sections.append(current)
            comment = []
            i += 3
            continue
        if line.startswith('//'):
            c = line[2:].strip()
            if not re.match(r'^/*$', c):
                comment.append(c)
        elif line.startswith('#'):
            pass
        else:
            m = re.match(r'^(?:unsigned int|int|double|bool|py::object)\s+(\w+);', line)
            if m and current is not None:
                current[1].append((m.group(1), ' '.join(comment)))
            if line == '' or m:
                comment = [] if m or line == '' else comment
        i += 1
    return sections


def main():
    p = NEAT.Parameters()
    out = [INTRO]
    for title, members in parse():
        rows = []
        for name, desc in members:
            if name in SKIP:
                continue
            pyname = RENAMED.get(name, name)
            desc = DESCRIPTIONS.get(pyname, desc)
            if not hasattr(p, pyname):
                continue
            default = getattr(p, pyname)
            if pyname in ('CustomConstraints', 'BehaviorGetter'):
                default = None
            if pyname == 'EliteFraction':
                desc += ' (also available as `Elitism`, the name used in parameter files)'
            rows.append('| `%s` | %s | %s |' % (pyname, fmt(default), desc.replace('|', '\\|')))
        if rows:
            out.append('\n## %s\n\n| Parameter | Default | Description |\n|---|---|---|' % title)
            out.extend(rows)
    open(OUT, 'w', encoding='utf8', newline='\n').write('\n'.join(out) + '\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
