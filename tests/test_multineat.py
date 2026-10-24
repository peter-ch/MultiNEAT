"""Tests for the MultiNEAT Python package.

Run with:  python -m pytest tests
"""
import locale
import math
import os
import pickle

import pytest

import MultiNEAT as NEAT


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def xor_params(pop_size=150):
    params = NEAT.Parameters()
    params.PopulationSize = pop_size
    params.DynamicCompatibility = True
    params.MinSpecies = 5
    params.MaxSpecies = 10
    params.RouletteWheelSelection = False
    params.TournamentSelection = False
    params.RecurrentProb = 0.0
    params.OverallMutationRate = 0.8
    params.MutateWeightsProb = 0.90
    params.WeightMutationMaxPower = 2.5
    params.WeightReplacementMaxPower = 5.0
    params.MutateWeightsSevereProb = 0.5
    params.WeightMutationRate = 0.25
    params.MaxWeight = 8
    params.MutateAddNeuronProb = 0.03
    params.MutateAddLinkProb = 0.05
    params.MutateRemLinkProb = 0.0
    params.MinActivationA = 4.9
    params.MaxActivationA = 4.9
    params.ActivationFunction_SignedSigmoid_Prob = 0.0
    params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
    params.CrossoverRate = 0.75
    params.MultipointCrossoverRate = 0.4
    params.SurvivalRate = 0.2
    params.AllowClones = True
    params.AllowLoops = False
    return params


def xor_genome(params):
    return NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                       NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0)


XOR_CASES = [([0, 0, 1], 0), ([0, 1, 1], 1), ([1, 0, 1], 1), ([1, 1, 1], 0)]


def evaluate_xor(genome):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)
    genome.CalculateDepth()
    depth = max(genome.GetDepth(), 1)
    error = 0.0
    for inputs, target in XOR_CASES:
        net.Flush()
        net.Input(inputs)
        for _ in range(depth):
            net.Activate()
        error += abs(net.Output()[0] - target)
    return (4 - error) ** 2


def run_generations(pop, evaluator, generations):
    best = -float('inf')
    for _ in range(generations):
        genomes = NEAT.GetGenomeList(pop)
        for g in genomes:
            f = evaluator(g)
            g.SetFitness(f)
            best = max(best, f)
        pop.Epoch()
    return best


def int_trait(min_, max_, prob=0.5):
    return {'details': {'max': max_, 'min': min_, 'mut_power': 1, 'mut_replace_prob': 0.1},
            'importance_coeff': 0.1, 'mutation_prob': prob, 'type': 'int'}


# ---------------------------------------------------------------------------
# basic API
# ---------------------------------------------------------------------------

def test_enums_and_parameters():
    assert int(NEAT.ActivationFunction.UNSIGNED_SIGMOID) == 1
    p = NEAT.Parameters()
    # GitHub #28 / #30 - these used to be missing
    p.MutateNeuronTraitsProb = 0.8
    p.MutateLinkTraitsProb = 0.8
    p.MutateGenomeTraitsProb = 0.8
    assert p.MutateNeuronTraitsProb == pytest.approx(0.8)
    # GitHub #54 - tournament selection is available
    p.TournamentSelection = True
    p.TournamentSize = 3
    assert p.TournamentSize == 3


def test_genome_constructors():
    params = NEAT.Parameters()
    # the old documented call without FS_NEAT_links (GitHub #58, #40)
    g = NEAT.Genome(0, 3, 0, 2, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0)
    assert g.NumInputs() == 3 and g.NumOutputs() == 2
    assert g.NumLinks() == 6

    # with the init struct
    gi = NEAT.GenomeInitStruct()
    gi.NumInputs = 4
    gi.NumOutputs = 2
    gi.NumHidden = 3
    gi.NumLayers = 1
    gi.SeedType = NEAT.GenomeSeedType.LAYERED
    g2 = NEAT.Genome(params, gi)
    assert g2.NumNeurons() == 4 + 2 + 3
    assert g2.NumLinks() == 4 * 3 + 3 * 2

    with pytest.raises(Exception):
        bad = NEAT.GenomeInitStruct()
        bad.NumInputs = 0
        NEAT.Genome(params, bad)


@pytest.mark.parametrize('links', [1, 2, 5, 100])
def test_fs_neat_terminates(links):
    # used to loop forever when more links were requested than possible
    params = NEAT.Parameters()
    g = NEAT.Genome(0, 3, 0, 2, True, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0, links)
    assert g.NumLinks() >= 2
    pairs = {(l.FromNeuronID, l.ToNeuronID) for l in g.LinkGenes}
    assert len(pairs) == g.NumLinks()  # no duplicate links


def test_population_is_live():
    params = xor_params(30)
    pop = NEAT.Population(xor_genome(params), params, True, 1.0, 0)
    assert pop.NumGenomes() == 30
    g = pop.Species[0].Individuals[0]
    g.SetFitness(42.0)
    # the change is visible through the population (reference semantics)
    assert pop.Species[0].Individuals[0].GetFitness() == 42.0
    assert pop.Species[0].GetLeader().GetFitness() == 42.0


def test_network_input_types():
    np = pytest.importorskip('numpy')
    params = NEAT.Parameters()
    g = xor_genome(params)
    net = NEAT.NeuralNetwork()
    g.BuildPhenotype(net)
    outs = []
    for inp in ([1.0, 0.0, 1.0], (1.0, 0.0, 1.0), np.array([1.0, 0.0, 1.0]),
                np.array([1, 0, 1], dtype=np.float32)):
        net.Flush()
        net.Input(inp)
        net.Activate()
        outs.append(net.Output()[0])
    assert all(o == pytest.approx(outs[0]) for o in outs)
    assert np.array(net.Output()).shape == (1,)


# ---------------------------------------------------------------------------
# evolution
# ---------------------------------------------------------------------------

def test_xor_is_solved():
    solved = 0
    for seed in range(3):
        params = xor_params()
        pop = NEAT.Population(xor_genome(params), params, True, 1.0, seed)
        best = 0
        for _ in range(250):
            for g in NEAT.GetGenomeList(pop):
                f = evaluate_xor(g)
                g.SetFitness(f)
                best = max(best, f)
            if best > 15.0:
                break
            pop.Epoch()
        solved += best > 15.0
    assert solved >= 2


def test_negative_and_nonfinite_fitness():
    # GitHub #26 - negative fitness used to trigger asserts
    params = xor_params(40)
    pop = NEAT.Population(xor_genome(params), params, True, 1.0, 1)
    values = [-5.0, -1e6, float('nan'), float('inf'), 0.0]
    for gen in range(15):
        for i, g in enumerate(NEAT.GetGenomeList(pop)):
            g.SetFitness(values[(i + gen) % len(values)] - i)
        pop.Epoch()
    assert pop.NumGenomes() >= params.PopulationSize


@pytest.mark.parametrize('selection', ['roulette', 'tournament', 'truncation'])
def test_selection_schemes(selection):
    # GitHub #54 - roulette selection used to crash
    params = xor_params(40)
    params.RouletteWheelSelection = selection == 'roulette'
    params.TournamentSelection = selection == 'tournament'
    params.TournamentSize = 4
    params.InterspeciesCrossoverRate = 0.2
    pop = NEAT.Population(xor_genome(params), params, True, 1.0, 2)
    best = run_generations(pop, evaluate_xor, 20)
    assert best > 0


def test_steady_state_tick():
    params = xor_params(30)
    pop = NEAT.Population(xor_genome(params), params, True, 1.0, 3)
    for g in NEAT.GetGenomeList(pop):
        g.SetFitness(evaluate_xor(g))
        g.SetEvaluated()
    for _ in range(100):
        removed = NEAT.Genome()
        baby = pop.Tick(removed)
        baby.SetFitness(evaluate_xor(baby))
        baby.SetEvaluated()
    assert pop.NumGenomes() == 30


def test_mutation_never_hangs_without_mutations():
    # all mutation probabilities 0 - this used to loop forever
    params = xor_params(20)
    for name in ['MutateAddNeuronProb', 'MutateAddLinkProb', 'MutateRemLinkProb', 'MutateRemSimpleNeuronProb',
                 'MutateNeuronActivationTypeProb', 'MutateWeightsProb', 'MutateActivationAProb',
                 'MutateActivationBProb', 'MutateNeuronTimeConstantsProb', 'MutateNeuronBiasesProb']:
        setattr(params, name, 0.0)
    params.CrossoverRate = 0.0
    pop = NEAT.Population(xor_genome(params), params, True, 1.0, 4)
    run_generations(pop, evaluate_xor, 3)


def test_novelty_search_with_python_behaviors():
    # GitHub #20 - behaviors are plain Python objects with a distance_to() method
    class Behavior:
        def __init__(self, v):
            self.v = v

        def distance_to(self, other):
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.v, other.v)))

    params = xor_params(30)
    pop = NEAT.Population(xor_genome(params), params, True, 1.0, 5)
    archive = []

    def behavior_of(genome):
        net = NEAT.NeuralNetwork()
        genome.BuildPhenotype(net)
        out = []
        for inputs, _ in XOR_CASES:
            net.Flush()
            net.Input(inputs)
            net.Activate()
            out.append(net.Output()[0])
        return Behavior(out)

    def sparseness(genome, k=5):
        others = [g.behavior for g in NEAT.GetGenomeList(pop)] + archive
        d = sorted(genome.behavior.distance_to(o) for o in others)
        return sum(d[1:k + 1]) / k

    for g in NEAT.GetGenomeList(pop):
        g.behavior = behavior_of(g)
    for g in NEAT.GetGenomeList(pop):
        g.SetFitness(sparseness(g))
        g.SetEvaluated()
    for _ in range(50):
        new = pop.Tick(NEAT.Genome())
        new.behavior = behavior_of(new)
        s = sparseness(new)
        new.SetFitness(s)
        new.SetEvaluated()
        if s > 0.1:
            archive.append(new.behavior)
    assert pop.NumGenomes() == 30


# ---------------------------------------------------------------------------
# traits
# ---------------------------------------------------------------------------

def make_trait_params():
    params = xor_params(30)
    params.MutateNeuronTraitsProb = 0.5
    params.MutateLinkTraitsProb = 0.5
    params.MutateGenomeTraitsProb = 0.5
    params.SetNeuronTraitParameters('n', int_trait(-5, 5))
    params.SetLinkTraitParameters('l', {'details': {'set': ['a', 'b', 'c'], 'probs': [1, 1, 1]},
                                        'importance_coeff': 0.1, 'mutation_prob': 0.5, 'type': 'str'})
    params.SetGenomeTraitParameters('g', {'details': {'min': 0.0, 'max': 1.0, 'mut_power': 0.1,
                                                      'mut_replace_prob': 0.1},
                                          'importance_coeff': 0.1, 'mutation_prob': 0.5, 'type': 'float'})
    # a trait with a single possible value can't be mutated - this used to hang
    params.SetGenomeTraitParameters('fixed', int_trait(3, 3, prob=1.0))
    return params


def test_traits_evolve_and_survive_pickle():
    params = make_trait_params()
    pop = NEAT.Population(xor_genome(params), params, True, 1.0, 6)
    run_generations(pop, evaluate_xor, 5)
    g = pop.GetBestGenome()
    gt = g.GetGenomeTraits()
    assert set(gt) == {'g', 'fixed'} and gt['fixed'] == 3
    assert all('l' in t for _, _, t in g.GetLinkTraits())

    # GitHub #41 - traits were lost when genomes were sent to other processes
    g2 = pickle.loads(pickle.dumps(g))
    assert g2.GetGenomeTraits() == gt
    assert g2.GetNeuronTraits() == g.GetNeuronTraits()
    assert g2.GetLinkTraits(True) == g.GetLinkTraits(True)

    p2 = pickle.loads(pickle.dumps(params))
    assert sorted(p2.ListNeuronTraitParameters()) == ['n']
    assert p2.GetGenomeTraitParameters('g')['type'] == 'float'


def test_dependent_traits():
    params = xor_params(20)
    params.MutateNeuronTraitsProb = 1.0
    params.SetNeuronTraitParameters('kind', {'details': {'set': ['x', 'y'], 'probs': [1, 1]},
                                             'importance_coeff': 0.0, 'mutation_prob': 0.5, 'type': 'str'})
    t = int_trait(0, 10)
    t['dep_key'] = 'kind'
    t['dep_values'] = ['y']
    params.SetNeuronTraitParameters('only_for_y', t)
    back = params.GetNeuronTraitParameters('only_for_y')
    assert back['dep_key'] == 'kind' and list(back['dep_values']) == ['y']
    pop = NEAT.Population(xor_genome(params), params, True, 1.0, 7)
    run_generations(pop, evaluate_xor, 5)
    for _, tp, traits in pop.GetBestGenome().GetNeuronTraits():
        if tp != 'input' and tp != 'bias':
            assert ('only_for_y' in traits) == (traits['kind'] == 'y')


# ---------------------------------------------------------------------------
# saving / loading / pickling
# ---------------------------------------------------------------------------

def test_pickle_population_and_continue():
    params = xor_params(40)
    pop = NEAT.Population(xor_genome(params), params, True, 1.0, 8)
    run_generations(pop, evaluate_xor, 3)
    pop2 = pickle.loads(pickle.dumps(pop))
    assert pop2.NumGenomes() == pop.NumGenomes()
    run_generations(pop2, evaluate_xor, 3)


def test_pickle_substrate_and_parameters():
    # GitHub #34
    s = NEAT.Substrate([(-1, -1), (1, -1)], [(0, 0)], [(0, 1)])
    s.m_leaky = True
    s2 = pickle.loads(pickle.dumps(s))
    assert s2.m_leaky and len(s2.m_input_coords) == 2 and list(s2.m_output_coords[0]) == [0, 1]
    p = NEAT.Parameters()
    p.PopulationSize = 123
    p.TournamentSelection = False
    p2 = pickle.loads(pickle.dumps(p))
    assert p2.PopulationSize == 123 and not p2.TournamentSelection


def test_parameters_save_load(tmp_path):
    p = NEAT.Parameters()
    p.PopulationSize = 77
    p.MutateNeuronTraitsProb = 0.25
    p.CompatTreshold = 1.2345
    fn = str(tmp_path / 'params.txt')
    p.Save(fn)
    q = NEAT.Parameters()
    assert q.Load(fn) == 0
    assert q.PopulationSize == 77
    assert q.MutateNeuronTraitsProb == pytest.approx(0.25)
    assert q.CompatTreshold == pytest.approx(1.2345)


def test_load_bad_files_does_not_hang(tmp_path):
    # GitHub #14, #64, #66 - the loaders used to loop forever on bad input
    junk = tmp_path / 'junk.txt'
    junk.write_text('this is not a valid file 1,5 2,5\n')
    truncated = tmp_path / 'truncated.txt'
    truncated.write_text('NEAT_ParametersStart\nPopulationSize 10\nCompatTreshold 0,5\n')
    missing = str(tmp_path / 'missing.txt')

    p = NEAT.Parameters()
    assert p.Load(str(junk)) != 0
    assert p.Load(str(truncated)) != 0
    assert p.Load(missing) != 0

    net = NEAT.NeuralNetwork()
    assert not net.Load(str(junk))
    assert not net.Load(missing)
    (tmp_path / 'nn.txt').write_text('NNstart\n3 1\nneuron 1 1,5 0 0 0 1 0\n')
    assert not net.Load(str(tmp_path / 'nn.txt'))

    with pytest.raises(Exception):
        NEAT.Genome(str(junk))
    (tmp_path / 'g.txt').write_text('GenomeStart 1\nNeuron 1 1 0,5 0 0 0 0 0\n')
    with pytest.raises(Exception):
        NEAT.Genome(str(tmp_path / 'g.txt'))


def test_genome_and_network_save_load(tmp_path):
    params = xor_params(30)
    pop = NEAT.Population(xor_genome(params), params, True, 1.0, 9)
    run_generations(pop, evaluate_xor, 10)
    g = pop.GetBestGenome()

    fn = str(tmp_path / 'genome.txt')
    g.Save(fn)
    g2 = NEAT.Genome(fn)
    assert g2.NumNeurons() == g.NumNeurons() and g2.NumLinks() == g.NumLinks()
    assert [l.Weight for l in g2.LinkGenes] == [l.Weight for l in g.LinkGenes]

    net = NEAT.NeuralNetwork()
    g.BuildPhenotype(net)
    nfn = str(tmp_path / 'net.txt')
    net.Save(nfn)
    net2 = NEAT.NeuralNetwork()
    assert net2.Load(nfn)
    assert len(net2.neurons) == len(net.neurons) and len(net2.connections) == len(net.connections)
    for inputs, _ in XOR_CASES:
        outs = []
        for n in (net, net2):
            n.Flush()
            n.Input(inputs)
            for _ in range(5):
                n.Activate()
            outs.append(n.Output()[0])
        assert outs[0] == pytest.approx(outs[1])


def test_population_save_load(tmp_path):
    params = xor_params(30)
    pop = NEAT.Population(xor_genome(params), params, True, 1.0, 10)
    run_generations(pop, evaluate_xor, 5)
    fn = str(tmp_path / 'pop.txt')
    pop.Save(fn)
    pop2 = NEAT.Population(fn)
    assert pop2.NumGenomes() == pop.NumGenomes()
    run_generations(pop2, evaluate_xor, 2)


def _comma_locale():
    for name in ('de_DE.UTF-8', 'de_DE.utf8', 'de_DE', 'German_Germany.1252', 'German'):
        try:
            locale.setlocale(locale.LC_NUMERIC, name)
            if locale.localeconv()['decimal_point'] == ',':
                return True
        except locale.Error:
            pass
    return False


def test_save_load_with_comma_locale(tmp_path):
    # GitHub #64 - files written with a ',' decimal separator could not be read back
    old = locale.setlocale(locale.LC_NUMERIC)
    if not _comma_locale():
        pytest.skip('no locale with a comma decimal separator available')
    try:
        p = NEAT.Parameters()
        p.CompatTreshold = 2.5
        fn = tmp_path / 'p.txt'
        p.Save(str(fn))
        assert '2.5' in fn.read_text()
        g = xor_genome(p)
        gfn = tmp_path / 'g.txt'
        g.Save(str(gfn))
        assert NEAT.Genome(str(gfn)).NumLinks() == g.NumLinks()
        q = NEAT.Parameters()
        assert q.Load(str(fn)) == 0 and q.CompatTreshold == pytest.approx(2.5)
    finally:
        locale.setlocale(locale.LC_NUMERIC, old)


# ---------------------------------------------------------------------------
# HyperNEAT / ES-HyperNEAT
# ---------------------------------------------------------------------------

def make_substrate(dims=2):
    pad = (0.0,) * (dims - 2)
    s = NEAT.Substrate([(-1., -1.) + pad, (1., -1.) + pad, (0., -1.) + pad],
                       [(0., 0.) + pad],
                       [(0., 1.) + pad])
    s.m_allow_input_hidden_links = True
    s.m_allow_hidden_output_links = True
    s.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID
    s.m_output_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID
    s.m_max_weight_and_bias = 8.0
    return s


def cppn_genome(params, substrate):
    return NEAT.Genome(0, substrate.GetMinCPPNInputs(), 0, substrate.GetMinCPPNOutputs(), False,
                       NEAT.ActivationFunction.TANH, NEAT.ActivationFunction.TANH, 0, params, 0)


def test_hyperneat_phenotype():
    s = make_substrate()
    params = NEAT.Parameters()
    params.PopulationSize = 20
    pop = NEAT.Population(cppn_genome(params, s), params, True, 1.0, 11)
    net = NEAT.NeuralNetwork()
    for g in NEAT.GetGenomeList(pop):
        g.BuildHyperNEATPhenotype(net, s)
        assert len(net.neurons) == 5
    # building twice into the same network must not accumulate neurons
    g.BuildHyperNEATPhenotype(net, s)
    assert len(net.neurons) == 5


@pytest.mark.parametrize('dims', [2, 3])
def test_es_hyperneat(tmp_path, dims):
    s = make_substrate(dims)
    params = NEAT.Parameters()
    params.PopulationSize = 30
    params.DivisionThreshold = 0.03
    params.VarianceThreshold = 0.03
    params.BandThreshold = 0.1
    params.InitialDepth = 2
    params.MaxDepth = 3
    params.IterationLevel = 1
    params.CPPN_Bias = 1.0
    params.Width = 1.0
    params.Height = 1.0
    pop = NEAT.Population(cppn_genome(params, s), params, True, 2.0, 12)
    found_hidden = False
    for i, g in enumerate(NEAT.GetGenomeList(pop)):
        net = NEAT.NeuralNetwork()
        g.BuildESHyperNEATPhenotype(net, s, params)
        assert net.NumInputs() == 3 and net.NumOutputs() == 1
        n = len(net.neurons)
        for c in net.connections:
            assert 0 <= c.source_neuron_idx < n and 0 <= c.target_neuron_idx < n
            assert net.neurons[c.target_neuron_idx].type != NEAT.NeuronType.INPUT
        found_hidden |= n > 4
        net.Input([1, 0, 1])
        net.Activate()
        # GitHub #66 - saving and loading an ES-HyperNEAT network used to hang
        fn = str(tmp_path / ('es%d.txt' % i))
        net.Save(fn)
        net2 = NEAT.NeuralNetwork()
        assert net2.Load(fn)
        assert len(net2.neurons) == n
    assert found_hidden


def test_parallel_evaluation_keeps_traits():
    # GitHub #41 - genomes evaluated in other processes must keep their traits
    from concurrent.futures import ProcessPoolExecutor
    params = make_trait_params()
    pop = NEAT.Population(xor_genome(params), params, True, 1.0, 13)
    genomes = NEAT.GetGenomeList(pop)[:8]
    with ProcessPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(genome_trait_count, genomes))
    assert results == [genome_trait_count(g) for g in genomes]
    assert all(r > 0 for r in results)


def genome_trait_count(genome):
    return (len(genome.GetGenomeTraits()) + sum(len(t) for _, _, t in genome.GetNeuronTraits())
            + sum(len(t) for _, _, t in genome.GetLinkTraits()))


def test_python_object_members_default_to_none():
    p = NEAT.Parameters()
    assert p.CustomConstraints is None and p.BehaviorGetter is None
    g = NEAT.Genome()
    assert g.behavior is None
    g.behavior = [1, 2]
    assert g.behavior == [1, 2]
    p.CustomConstraints = None
    assert p.CustomConstraints is None
