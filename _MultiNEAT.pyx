# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: infer_types=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: c_string_type=str
# cython: c_string_encoding=ascii

from libcpp.vector cimport vector
from libcpp cimport bool
from cython.operator cimport dereference as deref, preincrement as preinc
cimport cMultiNeat as cmn


cdef class RNG:
    cdef cmn.RNG* thisptr      # hold a C++ instance which we're wrapping
    cdef bint borrowed

    def __cinit__(self):
        self.thisptr = new cmn.RNG()
        self.borrowed = False

    def __dealloc__(self):
        if not self.borrowed:
            del self.thisptr

    def Seed(self, int a_seed):
        self.thisptr.Seed(a_seed)
    def TimeSeed(self):
        self.thisptr.TimeSeed()
    def RandPosNeg(self):
        return self.thisptr.RandPosNeg()
    def RandInt(self, x, y):
        return self.thisptr.RandInt(x, y)
    def RandFloat(self):
        return self.thisptr.RandFloat()
    def RandFloatClamped(self):
        return self.thisptr.RandFloatSigned()
    def RandGaussClamped(self):
        return self.thisptr.RandGaussSigned()
    def Roulette(self, a_probs):
        return self.thisptr.Roulette(a_probs)


cdef class Parameters:
    cdef cmn.Parameters *thisptr      # hold a C++ instance which we're wrapping
    cdef bint borrowed

    def __cinit__(self):
        self.thisptr = new cmn.Parameters()
        self.borrowed = False

    def __dealloc__(self):
        if not self.borrowed:
            del self.thisptr

    def Load(self, filename):
        self.thisptr.Load(filename)
    def Save(self, filename):
        self.thisptr.Save(filename)
    def Reset(self):
        self.thisptr.Reset()

    property PopulationSize:
        def __get__(self): return self.thisptr.PopulationSize
        def __set__(self, PopulationSize): self.thisptr.PopulationSize = PopulationSize
        
    property DynamicCompatibility:
        def __get__(self): return self.thisptr.DynamicCompatibility
        def __set__(self, DynamicCompatibility): self.thisptr.DynamicCompatibility = DynamicCompatibility
        
    property MinSpecies:
        def __get__(self): return self.thisptr.MinSpecies
        def __set__(self, MinSpecies): self.thisptr.MinSpecies = MinSpecies
        
    property MaxSpecies:
        def __get__(self): return self.thisptr.MaxSpecies
        def __set__(self, MaxSpecies): self.thisptr.MaxSpecies = MaxSpecies
        
    property InnovationsForever:
        def __get__(self): return self.thisptr.InnovationsForever
        def __set__(self, InnovationsForever): self.thisptr.InnovationsForever = InnovationsForever
        
    property AllowClones:
        def __get__(self): return self.thisptr.AllowClones
        def __set__(self, AllowClones): self.thisptr.AllowClones = AllowClones
        
    property YoungAgeTreshold:
        def __get__(self): return self.thisptr.YoungAgeTreshold
        def __set__(self, YoungAgeTreshold): self.thisptr.YoungAgeTreshold = YoungAgeTreshold
        
    property YoungAgeFitnessBoost:
        def __get__(self): return self.thisptr.YoungAgeFitnessBoost
        def __set__(self, YoungAgeFitnessBoost): self.thisptr.YoungAgeFitnessBoost = YoungAgeFitnessBoost
        
    property SpeciesMaxStagnation:
        def __get__(self): return self.thisptr.SpeciesMaxStagnation
        def __set__(self, SpeciesMaxStagnation): self.thisptr.SpeciesMaxStagnation = SpeciesMaxStagnation
        
    property StagnationDelta:
        def __get__(self): return self.thisptr.StagnationDelta
        def __set__(self, StagnationDelta): self.thisptr.StagnationDelta = StagnationDelta
        
    property OldAgeTreshold:
        def __get__(self): return self.thisptr.OldAgeTreshold
        def __set__(self, OldAgeTreshold): self.thisptr.OldAgeTreshold = OldAgeTreshold
        
    property OldAgePenalty:
        def __get__(self): return self.thisptr.OldAgePenalty
        def __set__(self, OldAgePenalty): self.thisptr.OldAgePenalty = OldAgePenalty
        
    property DetectCompetetiveCoevolutionStagnation:
        def __get__(self): return self.thisptr.DetectCompetetiveCoevolutionStagnation
        def __set__(self, DetectCompetetiveCoevolutionStagnation): self.thisptr.DetectCompetetiveCoevolutionStagnation = DetectCompetetiveCoevolutionStagnation
        
    property KillWorstSpeciesEach:
        def __get__(self): return self.thisptr.KillWorstSpeciesEach
        def __set__(self, KillWorstSpeciesEach): self.thisptr.KillWorstSpeciesEach = KillWorstSpeciesEach
        
    property KillWorstAge:
        def __get__(self): return self.thisptr.KillWorstAge
        def __set__(self, KillWorstAge): self.thisptr.KillWorstAge = KillWorstAge
        
    property SurvivalRate:
        def __get__(self): return self.thisptr.SurvivalRate
        def __set__(self, SurvivalRate): self.thisptr.SurvivalRate = SurvivalRate
        
    property CrossoverRate:
        def __get__(self): return self.thisptr.CrossoverRate
        def __set__(self, CrossoverRate): self.thisptr.CrossoverRate = CrossoverRate
        
    property OverallMutationRate:
        def __get__(self): return self.thisptr.OverallMutationRate
        def __set__(self, OverallMutationRate): self.thisptr.OverallMutationRate = OverallMutationRate
        
    property InterspeciesCrossoverRate:
        def __get__(self): return self.thisptr.InterspeciesCrossoverRate
        def __set__(self, InterspeciesCrossoverRate): self.thisptr.InterspeciesCrossoverRate = InterspeciesCrossoverRate
        
    property MultipointCrossoverRate:
        def __get__(self): return self.thisptr.MultipointCrossoverRate
        def __set__(self, MultipointCrossoverRate): self.thisptr.MultipointCrossoverRate = MultipointCrossoverRate
        
    property RouletteWheelSelection:
        def __get__(self): return self.thisptr.RouletteWheelSelection
        def __set__(self, RouletteWheelSelection): self.thisptr.RouletteWheelSelection = RouletteWheelSelection
        
    property PhasedSearching:
        def __get__(self): return self.thisptr.PhasedSearching
        def __set__(self, PhasedSearching): self.thisptr.PhasedSearching = PhasedSearching
        
    property DeltaCoding:
        def __get__(self): return self.thisptr.DeltaCoding
        def __set__(self, DeltaCoding): self.thisptr.DeltaCoding = DeltaCoding
        
    property SimplifyingPhaseMPCTreshold:
        def __get__(self): return self.thisptr.SimplifyingPhaseMPCTreshold
        def __set__(self, SimplifyingPhaseMPCTreshold): self.thisptr.SimplifyingPhaseMPCTreshold = SimplifyingPhaseMPCTreshold
        
    property SimplifyingPhaseStagnationTreshold:
        def __get__(self): return self.thisptr.SimplifyingPhaseStagnationTreshold
        def __set__(self, SimplifyingPhaseStagnationTreshold): self.thisptr.SimplifyingPhaseStagnationTreshold = SimplifyingPhaseStagnationTreshold
        
    property ComplexityFloorGenerations:
        def __get__(self): return self.thisptr.ComplexityFloorGenerations
        def __set__(self, ComplexityFloorGenerations): self.thisptr.ComplexityFloorGenerations = ComplexityFloorGenerations
        
    property NoveltySearch_K:
        def __get__(self): return self.thisptr.NoveltySearch_K
        def __set__(self, NoveltySearch_K): self.thisptr.NoveltySearch_K = NoveltySearch_K
        
    property NoveltySearch_P_min:
        def __get__(self): return self.thisptr.NoveltySearch_P_min
        def __set__(self, NoveltySearch_P_min): self.thisptr.NoveltySearch_P_min = NoveltySearch_P_min
        
    property NoveltySearch_Dynamic_Pmin:
        def __get__(self): return self.thisptr.NoveltySearch_Dynamic_Pmin
        def __set__(self, NoveltySearch_Dynamic_Pmin): self.thisptr.NoveltySearch_Dynamic_Pmin = NoveltySearch_Dynamic_Pmin
        
    property NoveltySearch_No_Archiving_Stagnation_Treshold:
        def __get__(self): return self.thisptr.NoveltySearch_No_Archiving_Stagnation_Treshold
        def __set__(self, NoveltySearch_No_Archiving_Stagnation_Treshold): self.thisptr.NoveltySearch_No_Archiving_Stagnation_Treshold = NoveltySearch_No_Archiving_Stagnation_Treshold
        
    property NoveltySearch_Pmin_lowering_multiplier:
        def __get__(self): return self.thisptr.NoveltySearch_Pmin_lowering_multiplier
        def __set__(self, NoveltySearch_Pmin_lowering_multiplier): self.thisptr.NoveltySearch_Pmin_lowering_multiplier = NoveltySearch_Pmin_lowering_multiplier
        
    property NoveltySearch_Pmin_min:
        def __get__(self): return self.thisptr.NoveltySearch_Pmin_min
        def __set__(self, NoveltySearch_Pmin_min): self.thisptr.NoveltySearch_Pmin_min = NoveltySearch_Pmin_min
        
    property NoveltySearch_Quick_Archiving_Min_Evaluations:
        def __get__(self): return self.thisptr.NoveltySearch_Quick_Archiving_Min_Evaluations
        def __set__(self, NoveltySearch_Quick_Archiving_Min_Evaluations): self.thisptr.NoveltySearch_Quick_Archiving_Min_Evaluations = NoveltySearch_Quick_Archiving_Min_Evaluations
        
    property NoveltySearch_Pmin_raising_multiplier:
        def __get__(self): return self.thisptr.NoveltySearch_Pmin_raising_multiplier
        def __set__(self, NoveltySearch_Pmin_raising_multiplier): self.thisptr.NoveltySearch_Pmin_raising_multiplier = NoveltySearch_Pmin_raising_multiplier
        
    property NoveltySearch_Recompute_Sparseness_Each:
        def __get__(self): return self.thisptr.NoveltySearch_Recompute_Sparseness_Each
        def __set__(self, NoveltySearch_Recompute_Sparseness_Each): self.thisptr.NoveltySearch_Recompute_Sparseness_Each = NoveltySearch_Recompute_Sparseness_Each
        
    property MutateAddNeuronProb:
        def __get__(self): return self.thisptr.MutateAddNeuronProb
        def __set__(self, MutateAddNeuronProb): self.thisptr.MutateAddNeuronProb = MutateAddNeuronProb
        
    property SplitRecurrent:
        def __get__(self): return self.thisptr.SplitRecurrent
        def __set__(self, SplitRecurrent): self.thisptr.SplitRecurrent = SplitRecurrent
        
    property SplitLoopedRecurrent:
        def __get__(self): return self.thisptr.SplitLoopedRecurrent
        def __set__(self, SplitLoopedRecurrent): self.thisptr.SplitLoopedRecurrent = SplitLoopedRecurrent
        
    property NeuronTries:
        def __get__(self): return self.thisptr.NeuronTries
        def __set__(self, NeuronTries): self.thisptr.NeuronTries = NeuronTries
        
    property MutateAddLinkProb:
        def __get__(self): return self.thisptr.MutateAddLinkProb
        def __set__(self, MutateAddLinkProb): self.thisptr.MutateAddLinkProb = MutateAddLinkProb
        
    property MutateAddLinkFromBiasProb:
        def __get__(self): return self.thisptr.MutateAddLinkFromBiasProb
        def __set__(self, MutateAddLinkFromBiasProb): self.thisptr.MutateAddLinkFromBiasProb = MutateAddLinkFromBiasProb
        
    property MutateRemLinkProb:
        def __get__(self): return self.thisptr.MutateRemLinkProb
        def __set__(self, MutateRemLinkProb): self.thisptr.MutateRemLinkProb = MutateRemLinkProb
        
    property MutateRemSimpleNeuronProb:
        def __get__(self): return self.thisptr.MutateRemSimpleNeuronProb
        def __set__(self, MutateRemSimpleNeuronProb): self.thisptr.MutateRemSimpleNeuronProb = MutateRemSimpleNeuronProb
        
    property LinkTries:
        def __get__(self): return self.thisptr.LinkTries
        def __set__(self, LinkTries): self.thisptr.LinkTries = LinkTries
        
    property RecurrentProb:
        def __get__(self): return self.thisptr.RecurrentProb
        def __set__(self, RecurrentProb): self.thisptr.RecurrentProb = RecurrentProb
        
    property RecurrentLoopProb:
        def __get__(self): return self.thisptr.RecurrentLoopProb
        def __set__(self, RecurrentLoopProb): self.thisptr.RecurrentLoopProb = RecurrentLoopProb
        
    property MutateWeightsProb:
        def __get__(self): return self.thisptr.MutateWeightsProb
        def __set__(self, MutateWeightsProb): self.thisptr.MutateWeightsProb = MutateWeightsProb
        
    property MutateWeightsSevereProb:
        def __get__(self): return self.thisptr.MutateWeightsSevereProb
        def __set__(self, MutateWeightsSevereProb): self.thisptr.MutateWeightsSevereProb = MutateWeightsSevereProb
        
    property WeightMutationRate:
        def __get__(self): return self.thisptr.WeightMutationRate
        def __set__(self, WeightMutationRate): self.thisptr.WeightMutationRate = WeightMutationRate
        
    property WeightMutationMaxPower:
        def __get__(self): return self.thisptr.WeightMutationMaxPower
        def __set__(self, WeightMutationMaxPower): self.thisptr.WeightMutationMaxPower = WeightMutationMaxPower
        
    property WeightReplacementMaxPower:
        def __get__(self): return self.thisptr.WeightReplacementMaxPower
        def __set__(self, WeightReplacementMaxPower): self.thisptr.WeightReplacementMaxPower = WeightReplacementMaxPower
        
    property MaxWeight:
        def __get__(self): return self.thisptr.MaxWeight
        def __set__(self, MaxWeight): self.thisptr.MaxWeight = MaxWeight
        
    property MutateActivationAProb:
        def __get__(self): return self.thisptr.MutateActivationAProb
        def __set__(self, MutateActivationAProb): self.thisptr.MutateActivationAProb = MutateActivationAProb
        
    property MutateActivationBProb:
        def __get__(self): return self.thisptr.MutateActivationBProb
        def __set__(self, MutateActivationBProb): self.thisptr.MutateActivationBProb = MutateActivationBProb
        
    property ActivationAMutationMaxPower:
        def __get__(self): return self.thisptr.ActivationAMutationMaxPower
        def __set__(self, ActivationAMutationMaxPower): self.thisptr.ActivationAMutationMaxPower = ActivationAMutationMaxPower
        
    property ActivationBMutationMaxPower:
        def __get__(self): return self.thisptr.ActivationBMutationMaxPower
        def __set__(self, ActivationBMutationMaxPower): self.thisptr.ActivationBMutationMaxPower = ActivationBMutationMaxPower
        
    property TimeConstantMutationMaxPower:
        def __get__(self): return self.thisptr.TimeConstantMutationMaxPower
        def __set__(self, TimeConstantMutationMaxPower): self.thisptr.TimeConstantMutationMaxPower = TimeConstantMutationMaxPower
        
    property BiasMutationMaxPower:
        def __get__(self): return self.thisptr.BiasMutationMaxPower
        def __set__(self, BiasMutationMaxPower): self.thisptr.BiasMutationMaxPower = BiasMutationMaxPower
        
    property MinActivationA:
        def __get__(self): return self.thisptr.MinActivationA
        def __set__(self, MinActivationA): self.thisptr.MinActivationA = MinActivationA
        
    property MaxActivationA:
        def __get__(self): return self.thisptr.MaxActivationA
        def __set__(self, MaxActivationA): self.thisptr.MaxActivationA = MaxActivationA
        
    property MinActivationB:
        def __get__(self): return self.thisptr.MinActivationB
        def __set__(self, MinActivationB): self.thisptr.MinActivationB = MinActivationB
        
    property MaxActivationB:
        def __get__(self): return self.thisptr.MaxActivationB
        def __set__(self, MaxActivationB): self.thisptr.MaxActivationB = MaxActivationB
        
    property MutateNeuronActivationTypeProb:
        def __get__(self): return self.thisptr.MutateNeuronActivationTypeProb
        def __set__(self, MutateNeuronActivationTypeProb): self.thisptr.MutateNeuronActivationTypeProb = MutateNeuronActivationTypeProb
        
    property ActivationFunction_SignedSigmoid_Prob:
        def __get__(self): return self.thisptr.ActivationFunction_SignedSigmoid_Prob
        def __set__(self, ActivationFunction_SignedSigmoid_Prob): self.thisptr.ActivationFunction_SignedSigmoid_Prob = ActivationFunction_SignedSigmoid_Prob
        
    property ActivationFunction_UnsignedSigmoid_Prob:
        def __get__(self): return self.thisptr.ActivationFunction_UnsignedSigmoid_Prob
        def __set__(self, ActivationFunction_UnsignedSigmoid_Prob): self.thisptr.ActivationFunction_UnsignedSigmoid_Prob = ActivationFunction_UnsignedSigmoid_Prob
        
    property ActivationFunction_Tanh_Prob:
        def __get__(self): return self.thisptr.ActivationFunction_Tanh_Prob
        def __set__(self, ActivationFunction_Tanh_Prob): self.thisptr.ActivationFunction_Tanh_Prob = ActivationFunction_Tanh_Prob
        
    property ActivationFunction_TanhCubic_Prob:
        def __get__(self): return self.thisptr.ActivationFunction_TanhCubic_Prob
        def __set__(self, ActivationFunction_TanhCubic_Prob): self.thisptr.ActivationFunction_TanhCubic_Prob = ActivationFunction_TanhCubic_Prob
        
    property ActivationFunction_SignedStep_Prob:
        def __get__(self): return self.thisptr.ActivationFunction_SignedStep_Prob
        def __set__(self, ActivationFunction_SignedStep_Prob): self.thisptr.ActivationFunction_SignedStep_Prob = ActivationFunction_SignedStep_Prob
        
    property ActivationFunction_UnsignedStep_Prob:
        def __get__(self): return self.thisptr.ActivationFunction_UnsignedStep_Prob
        def __set__(self, ActivationFunction_UnsignedStep_Prob): self.thisptr.ActivationFunction_UnsignedStep_Prob = ActivationFunction_UnsignedStep_Prob
        
    property ActivationFunction_SignedGauss_Prob:
        def __get__(self): return self.thisptr.ActivationFunction_SignedGauss_Prob
        def __set__(self, ActivationFunction_SignedGauss_Prob): self.thisptr.ActivationFunction_SignedGauss_Prob = ActivationFunction_SignedGauss_Prob
        
    property ActivationFunction_UnsignedGauss_Prob:
        def __get__(self): return self.thisptr.ActivationFunction_UnsignedGauss_Prob
        def __set__(self, ActivationFunction_UnsignedGauss_Prob): self.thisptr.ActivationFunction_UnsignedGauss_Prob = ActivationFunction_UnsignedGauss_Prob
        
    property ActivationFunction_Abs_Prob:
        def __get__(self): return self.thisptr.ActivationFunction_Abs_Prob
        def __set__(self, ActivationFunction_Abs_Prob): self.thisptr.ActivationFunction_Abs_Prob = ActivationFunction_Abs_Prob
        
    property ActivationFunction_SignedSine_Prob:
        def __get__(self): return self.thisptr.ActivationFunction_SignedSine_Prob
        def __set__(self, ActivationFunction_SignedSine_Prob): self.thisptr.ActivationFunction_SignedSine_Prob = ActivationFunction_SignedSine_Prob
        
    property ActivationFunction_UnsignedSine_Prob:
        def __get__(self): return self.thisptr.ActivationFunction_UnsignedSine_Prob
        def __set__(self, ActivationFunction_UnsignedSine_Prob): self.thisptr.ActivationFunction_UnsignedSine_Prob = ActivationFunction_UnsignedSine_Prob
        
    # property ActivationFunction_SignedSquare_Prob:
    #     def __get__(self): return self.thisptr.ActivationFunction_SignedSquare_Prob
    #     def __set__(self, ActivationFunction_SignedSquare_Prob): self.thisptr.ActivationFunction_SignedSquare_Prob = ActivationFunction_SignedSquare_Prob
    #
    # property ActivationFunction_UnsignedSquare_Prob:
    #     def __get__(self): return self.thisptr.ActivationFunction_UnsignedSquare_Prob
    #     def __set__(self, ActivationFunction_UnsignedSquare_Prob): self.thisptr.ActivationFunction_UnsignedSquare_Prob = ActivationFunction_UnsignedSquare_Prob
        
    property ActivationFunction_Linear_Prob:
        def __get__(self): return self.thisptr.ActivationFunction_Linear_Prob
        def __set__(self, ActivationFunction_Linear_Prob): self.thisptr.ActivationFunction_Linear_Prob = ActivationFunction_Linear_Prob
        
    property MutateNeuronTimeConstantsProb:
        def __get__(self): return self.thisptr.MutateNeuronTimeConstantsProb
        def __set__(self, MutateNeuronTimeConstantsProb): self.thisptr.MutateNeuronTimeConstantsProb = MutateNeuronTimeConstantsProb
        
    property MutateNeuronBiasesProb:
        def __get__(self): return self.thisptr.MutateNeuronBiasesProb
        def __set__(self, MutateNeuronBiasesProb): self.thisptr.MutateNeuronBiasesProb = MutateNeuronBiasesProb
        
    property MinNeuronTimeConstant:
        def __get__(self): return self.thisptr.MinNeuronTimeConstant
        def __set__(self, MinNeuronTimeConstant): self.thisptr.MinNeuronTimeConstant = MinNeuronTimeConstant
        
    property MaxNeuronTimeConstant:
        def __get__(self): return self.thisptr.MaxNeuronTimeConstant
        def __set__(self, MaxNeuronTimeConstant): self.thisptr.MaxNeuronTimeConstant = MaxNeuronTimeConstant
        
    property MinNeuronBias:
        def __get__(self): return self.thisptr.MinNeuronBias
        def __set__(self, MinNeuronBias): self.thisptr.MinNeuronBias = MinNeuronBias
        
    property MaxNeuronBias:
        def __get__(self): return self.thisptr.MaxNeuronBias
        def __set__(self, MaxNeuronBias): self.thisptr.MaxNeuronBias = MaxNeuronBias

    property DontUseBiasNeuron:
        def __get__(self): return self.thisptr.DontUseBiasNeuron
        def __set__(self, DontUseBiasNeuron): self.thisptr.DontUseBiasNeuron = DontUseBiasNeuron

    property AllowLoops:
        def __get__(self): return self.thisptr.AllowLoops
        def __set__(self, AllowLoops): self.thisptr.AllowLoops = AllowLoops

    property DisjointCoeff:
        def __get__(self): return self.thisptr.DisjointCoeff
        def __set__(self, DisjointCoeff): self.thisptr.DisjointCoeff = DisjointCoeff
        
    property ExcessCoeff:
        def __get__(self): return self.thisptr.ExcessCoeff
        def __set__(self, ExcessCoeff): self.thisptr.ExcessCoeff = ExcessCoeff
        
    property ActivationADiffCoeff:
        def __get__(self): return self.thisptr.ActivationADiffCoeff
        def __set__(self, ActivationADiffCoeff): self.thisptr.ActivationADiffCoeff = ActivationADiffCoeff
        
    property ActivationBDiffCoeff:
        def __get__(self): return self.thisptr.ActivationBDiffCoeff
        def __set__(self, ActivationBDiffCoeff): self.thisptr.ActivationBDiffCoeff = ActivationBDiffCoeff
        
    property WeightDiffCoeff:
        def __get__(self): return self.thisptr.WeightDiffCoeff
        def __set__(self, WeightDiffCoeff): self.thisptr.WeightDiffCoeff = WeightDiffCoeff
        
    property TimeConstantDiffCoeff:
        def __get__(self): return self.thisptr.TimeConstantDiffCoeff
        def __set__(self, TimeConstantDiffCoeff): self.thisptr.TimeConstantDiffCoeff = TimeConstantDiffCoeff
        
    property BiasDiffCoeff:
        def __get__(self): return self.thisptr.BiasDiffCoeff
        def __set__(self, BiasDiffCoeff): self.thisptr.BiasDiffCoeff = BiasDiffCoeff
        
    property ActivationFunctionDiffCoeff:
        def __get__(self): return self.thisptr.ActivationFunctionDiffCoeff
        def __set__(self, ActivationFunctionDiffCoeff): self.thisptr.ActivationFunctionDiffCoeff = ActivationFunctionDiffCoeff
        
    property CompatTreshold:
        def __get__(self): return self.thisptr.CompatTreshold
        def __set__(self, CompatTreshold): self.thisptr.CompatTreshold = CompatTreshold
        
    property MinCompatTreshold:
        def __get__(self): return self.thisptr.MinCompatTreshold
        def __set__(self, MinCompatTreshold): self.thisptr.MinCompatTreshold = MinCompatTreshold
        
    property CompatTresholdModifier:
        def __get__(self): return self.thisptr.CompatTresholdModifier
        def __set__(self, CompatTresholdModifier): self.thisptr.CompatTresholdModifier = CompatTresholdModifier
        
    property CompatTreshChangeInterval_Generations:
        def __get__(self): return self.thisptr.CompatTreshChangeInterval_Generations
        def __set__(self, CompatTreshChangeInterval_Generations): self.thisptr.CompatTreshChangeInterval_Generations = CompatTreshChangeInterval_Generations
        
    property CompatTreshChangeInterval_Evaluations:
        def __get__(self): return self.thisptr.CompatTreshChangeInterval_Evaluations
        def __set__(self, CompatTreshChangeInterval_Evaluations): self.thisptr.CompatTreshChangeInterval_Evaluations = CompatTreshChangeInterval_Evaluations

    property Elitism:
        '''Fraction of individuals to be copied unchanged'''
        def __get__(self): return self.thisptr.Elitism
        def __set__(self, double Elitism): self.thisptr.Elitism = Elitism

    ##############
    # ES HyperNEAT params
    ##############
    property DivisionThreshold:
        def __get__(self): return self.thisptr.DivisionThreshold
        def __set__(self, DivisionThreshold): self.thisptr.DivisionThreshold = DivisionThreshold

    property VarianceThreshold:
        def __get__(self): return self.thisptr.VarianceThreshold
        def __set__(self, VarianceThreshold): self.thisptr.VarianceThreshold = VarianceThreshold

    property BandThreshold:
        '''Used for Band prunning.'''
        def __get__(self): return self.thisptr.BandThreshold
        def __set__(self, BandThreshold): self.thisptr.BandThreshold = BandThreshold
        
    property InitialDepth:
        '''Min Depth of the quadtree'''
        def __get__(self): return self.thisptr.InitialDepth
        def __set__(self, unsigned int InitialDepth): self.thisptr.InitialDepth = InitialDepth

    property MaxDepth:
        '''Max Depth of the quadtree'''
        def __get__(self): return self.thisptr.MaxDepth
        def __set__(self, unsigned int MaxDepth): self.thisptr.MaxDepth = MaxDepth

    property IterationLevel:
        '''How many hidden layers before connecting nodes to output. 
        At 0 there is one hidden layer. At 1, there are two and so on.'''
        def __get__(self): return self.thisptr.IterationLevel
        def __set__(self, unsigned int IterationLevel): self.thisptr.IterationLevel = IterationLevel

    property CPPN_Bias:
        '''The Bias value for the CPPN queries'''
        def __get__(self): return self.thisptr.CPPN_Bias
        def __set__(self, double CPPN_Bias): self.thisptr.CPPN_Bias = CPPN_Bias

    property Width:
        def __get__(self): return self.thisptr.Width
        def __set__(self, double Width): self.thisptr.Width = Width

    property Height:
        def __get__(self): return self.thisptr.Height
        def __set__(self, double Height): self.thisptr.Height = Height

    property Qtree_X:
        def __get__(self): return self.thisptr.Qtree_X
        def __set__(self, double Qtree_X): self.thisptr.Qtree_X = Qtree_X

    property Qtree_Y:
        def __get__(self): return self.thisptr.Qtree_Y
        def __set__(self, double Qtree_Y): self.thisptr.Qtree_Y = Qtree_Y

    property Leo:
        '''Use Link Expression output'''
        def __get__(self): return self.thisptr.Leo
        def __set__(self, bint Leo): self.thisptr.Leo = Leo    

    property LeoThreshold:
        '''Threshold above which a connection is expressed'''
        def __get__(self): return self.thisptr.LeoThreshold
        def __set__(self, double LeoThreshold): self.thisptr.LeoThreshold = LeoThreshold

    property LeoSeed:
        def __get__(self): return self.thisptr.LeoSeed
        def __set__(self, bint LeoSeed): self.thisptr.LeoSeed = LeoSeed

    property GeometrySeed:
        def __get__(self): return self.thisptr.GeometrySeed
        def __set__(self, bint GeometrySeed): self.thisptr.GeometrySeed = GeometrySeed

cdef class Connection:
    cdef cmn.Connection *thisptr      # hold a C++ instance which we're wrapping
    cdef bint borrowed
    def __cinit__(self):
        self.thisptr = new cmn.Connection()
        self.borrowed = False
    def __dealloc__(self):
        if not self.borrowed:
            del self.thisptr

    property source_neuron_idx:
            def __get__(self): return self.thisptr.m_source_neuron_idx
            def __set__(self, m_source_neuron_idx): self.thisptr.m_source_neuron_idx = m_source_neuron_idx
            
    property target_neuron_idx:
            def __get__(self): return self.thisptr.m_target_neuron_idx
            def __set__(self, m_target_neuron_idx): self.thisptr.m_target_neuron_idx = m_target_neuron_idx
            
    property weight:
            def __get__(self): return self.thisptr.m_weight
            def __set__(self, m_weight): self.thisptr.m_weight = m_weight
            
    property recur_flag:
            def __get__(self): return self.thisptr.m_recur_flag
            def __set__(self, m_recur_flag): self.thisptr.m_recur_flag = m_recur_flag
            
            
cdef class Neuron:
    cdef cmn.Neuron *thisptr      # hold a C++ instance which we're wrapping
    cdef bint borrowed
    def __cinit__(self):
        self.thisptr = new cmn.Neuron()
        self.borrowed = False
    def __dealloc__(self):
        if not self.borrowed:
            del self.thisptr

    property activation:
            def __get__(self): return self.thisptr.m_activation
            def __set__(self, m_activation): self.thisptr.m_activation = m_activation
            
    property activation_function_type:
            def __get__(self): return self.thisptr.m_activation_function_type
            def __set__(self, m_activation_function_type): self.thisptr.m_activation_function_type = m_activation_function_type
            
    property a:
            def __get__(self): return self.thisptr.m_a
            def __set__(self, m_a): self.thisptr.m_a = m_a
            
    property b:
            def __get__(self): return self.thisptr.m_b
            def __set__(self, m_b): self.thisptr.m_b = m_b
            
    property timeconst:
            def __get__(self): return self.thisptr.m_timeconst
            def __set__(self, m_timeconst): self.thisptr.m_timeconst = m_timeconst
            
    property bias:
            def __get__(self): return self.thisptr.m_bias
            def __set__(self, m_bias): self.thisptr.m_bias = m_bias
            
    property x:
            def __get__(self): return self.thisptr.m_x
            def __set__(self, m_x): self.thisptr.m_x = m_x
            
    property y:
            def __get__(self): return self.thisptr.m_y
            def __set__(self, m_y): self.thisptr.m_y = m_y
            
    property z:
            def __get__(self): return self.thisptr.m_z
            def __set__(self, m_z): self.thisptr.m_z = m_z
            
    property sx:
            def __get__(self): return self.thisptr.m_sx
            def __set__(self, m_sx): self.thisptr.m_sx = m_sx
            
    property sy:
            def __get__(self): return self.thisptr.m_sy
            def __set__(self, m_sy): self.thisptr.m_sy = m_sy
            
    property sz:
            def __get__(self): return self.thisptr.m_sz
            def __set__(self, m_sz): self.thisptr.m_sz = m_sz
            
    property substrate_coords:
            def __get__(self): return self.thisptr.m_substrate_coords
            def __set__(self, m_substrate_coords): self.thisptr.m_substrate_coords = m_substrate_coords
            
    property split_y:
            def __get__(self): return self.thisptr.m_split_y
            def __set__(self, m_split_y): self.thisptr.m_split_y = m_split_y
            
    property type:
            def __get__(self): return self.thisptr.m_type
            def __set__(self, m_type): self.thisptr.m_type = m_type


cdef class NeuralNetwork:
    cdef cmn.NeuralNetwork *thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, x=None):
        if x is None:
            self.thisptr = new cmn.NeuralNetwork()
        else:
            self.thisptr = new cmn.NeuralNetwork(x)

    def __dealloc__(self):
        del self.thisptr

    def InitRTRLMatrix(self):
        self.thisptr.InitRTRLMatrix()
    
    def ActivateFast(self):
        self.thisptr.ActivateFast()
    
    def Activate(self):
        self.thisptr.Activate()
    
    def ActivateUseInternalBias(self):
        self.thisptr.ActivateUseInternalBias()
    
    def ActivateLeaky(self, step):
        self.thisptr.ActivateLeaky(step)
    
    def RTRL_update_gradients(self):
        self.thisptr.RTRL_update_gradients()
    
    def RTRL_update_error(self, double a_target):
        self.thisptr.RTRL_update_error(a_target)
    
    def RTRL_update_weights(self):
        self.thisptr.RTRL_update_weights()
    
    def Adapt(self, Parameters a_Parameters):
        self.thisptr.Adapt(deref(a_Parameters.thisptr))
    
    def Flush(self):
        self.thisptr.Flush()
    
    def FlushCube(self):
        self.thisptr.FlushCube()
    
    def Input(self, a_Inputs):
        self.thisptr.Input(a_Inputs)
    
    def Output(self):
        return self.thisptr.Output()

    def Clear(self):
        self.thisptr.Clear()

    def Save(self, const char* a_filename):
        self.thisptr.Save(a_filename)

    def Load(self, const char* a_filename):
        return self.thisptr.Load(a_filename)

    def NumInputs(self):
        return self.thisptr.NumInputs()

    def NumOutputs(self):
        return self.thisptr.NumOutputs()

    def AddNeuron(self, Neuron a_n):
        self.thisptr.AddNeuron(deref(a_n.thisptr))

    def AddConnection(self, Connection a_c):
        self.thisptr.AddConnection(deref(a_c.thisptr))

    def GetConnectionByIndex(self, unsigned int a_idx):
        cdef cmn.Connection ncon= self.thisptr.GetConnectionByIndex(a_idx)
        return pyConnectionFromReference(ncon)

    def GetNeuronByIndex(self, unsigned int a_idx):
        cdef cmn.Neuron nneur = self.thisptr.GetNeuronByIndex(a_idx)
        return pyNeuronFromReference(nneur)

    def SetInputOutputDimentions(self, const unsigned short a_i, const unsigned short a_o):
        self.thisptr.SetInputOutputDimentions(a_i, a_o)

    property num_inputs:
            def __get__(self): return self.thisptr.m_num_inputs
            def __set__(self, m_num_inputs): self.thisptr.m_num_inputs = m_num_inputs
            
    property num_outputs:
            def __get__(self): return self.thisptr.m_num_outputs
            def __set__(self, m_num_outputs): self.thisptr.m_num_outputs = m_num_outputs
            
    property neurons:
        def __get__(self): return neuronsVectorToList(self.thisptr.m_neurons)
        def __set__(self, list m_neurons): self.thisptr.m_neurons = neuronsListToVector(m_neurons)
            
    property connections:
           def __get__(self): return connectionsVectorToList(self.thisptr.m_connections)
           def __set__(self, list m_connections): self.thisptr.m_connections = connectionsListToVector(m_connections)
"""
#############################################

Substrate class

#############################################
"""
#cdef vector[ vector[double] ] x
#cdef vector[double] y
#y.push_back(1)
#x.push_back(y)


cdef class Substrate:
    cdef cmn.Substrate *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, *attribs):
        if len(attribs) == 0:
            self.thisptr = new cmn.Substrate()
        else:
            i, h, o = attribs
            self.thisptr = new cmn.Substrate(i, h, o)
    def __dealloc__(self):
        del self.thisptr
        
    def GetMinCPPNInputs(self):
        return self.thisptr.GetMinCPPNInputs()
    
    def GetMinCPPNOutputs(self):
        return self.thisptr.GetMinCPPNOutputs()
    
    def PrintInfo(self):
        return self.thisptr.PrintInfo()

    property m_input_coords:
            def __get__(self): return self.thisptr.m_input_coords
            def __set__(self, m_input_coords): self.thisptr.m_input_coords = m_input_coords
            
    property m_hidden_coords:
            def __get__(self): return self.thisptr.m_hidden_coords
            def __set__(self, m_hidden_coords): self.thisptr.m_hidden_coords = m_hidden_coords
            
    property m_output_coords:
            def __get__(self): return self.thisptr.m_output_coords
            def __set__(self, m_output_coords): self.thisptr.m_output_coords = m_output_coords
            
    property m_leaky:
            def __get__(self): return self.thisptr.m_leaky
            def __set__(self, m_leaky): self.thisptr.m_leaky = m_leaky
            
    property m_with_distance:
            def __get__(self): return self.thisptr.m_with_distance
            def __set__(self, m_with_distance): self.thisptr.m_with_distance = m_with_distance
            
    property m_allow_input_hidden_links:
            def __get__(self): return self.thisptr.m_allow_input_hidden_links
            def __set__(self, m_allow_input_hidden_links): self.thisptr.m_allow_input_hidden_links = m_allow_input_hidden_links
            
    property m_allow_input_output_links:
            def __get__(self): return self.thisptr.m_allow_input_output_links
            def __set__(self, m_allow_input_output_links): self.thisptr.m_allow_input_output_links = m_allow_input_output_links
            
    property m_allow_hidden_hidden_links:
            def __get__(self): return self.thisptr.m_allow_hidden_hidden_links
            def __set__(self, m_allow_hidden_hidden_links): self.thisptr.m_allow_hidden_hidden_links = m_allow_hidden_hidden_links
            
    property m_allow_hidden_output_links:
            def __get__(self): return self.thisptr.m_allow_hidden_output_links
            def __set__(self, m_allow_hidden_output_links): self.thisptr.m_allow_hidden_output_links = m_allow_hidden_output_links
            
    property m_allow_output_hidden_links:
            def __get__(self): return self.thisptr.m_allow_output_hidden_links
            def __set__(self, m_allow_output_hidden_links): self.thisptr.m_allow_output_hidden_links = m_allow_output_hidden_links
            
    property m_allow_output_output_links:
            def __get__(self): return self.thisptr.m_allow_output_output_links
            def __set__(self, m_allow_output_output_links): self.thisptr.m_allow_output_output_links = m_allow_output_output_links
            
    property m_allow_looped_hidden_links:
            def __get__(self): return self.thisptr.m_allow_looped_hidden_links
            def __set__(self, m_allow_looped_hidden_links): self.thisptr.m_allow_looped_hidden_links = m_allow_looped_hidden_links
            
    property m_allow_looped_output_links:
            def __get__(self): return self.thisptr.m_allow_looped_output_links
            def __set__(self, m_allow_looped_output_links): self.thisptr.m_allow_looped_output_links = m_allow_looped_output_links
            
    property m_hidden_nodes_activation:
            def __get__(self): return self.thisptr.m_hidden_nodes_activation
            def __set__(self, m_hidden_nodes_activation): self.thisptr.m_hidden_nodes_activation = m_hidden_nodes_activation
            
    property m_output_nodes_activation:
            def __get__(self): return self.thisptr.m_output_nodes_activation
            def __set__(self, m_output_nodes_activation): self.thisptr.m_output_nodes_activation = m_output_nodes_activation
            
    # property m_link_threshold:
    #         def __get__(self): return self.thisptr.m_link_threshold
    #         def __set__(self, m_link_threshold): self.thisptr.m_link_threshold = m_link_threshold
            
    property m_max_weight_and_bias:
            def __get__(self): return self.thisptr.m_max_weight_and_bias
            def __set__(self, m_max_weight_and_bias): self.thisptr.m_max_weight_and_bias = m_max_weight_and_bias
            
    property m_min_time_const:
            def __get__(self): return self.thisptr.m_min_time_const
            def __set__(self, m_min_time_const): self.thisptr.m_min_time_const = m_min_time_const
            
    property m_max_time_const:
            def __get__(self): return self.thisptr.m_max_time_const
            def __set__(self, m_max_time_const): self.thisptr.m_max_time_const = m_max_time_const


cdef class Genome:
    cdef cmn.Genome *thisptr      # hold a C++ instance which we're wrapping
    cdef bint borrowed

    def __cinit__(self, *attribs):
        cdef int attribsLen = len(attribs)
        cdef Parameters a_ps
        self.borrowed = False
        if attribsLen == 0:
            self.thisptr = new cmn.Genome()
        elif attribsLen == 1:
            if isinstance(attribs[0], Genome):
                self.thisptr = new cmn.Genome(deref(<cmn.Genome*>((<Genome>attribs[0]).thisptr)))
            elif isinstance(attribs[0], str):
                self.thisptr = new cmn.Genome(<char*>attribs[0])
        elif attribsLen == 9:
            a_id, a_ni, a_nh, a_no, a_fs, a_oa, a_ha, a_st, a_ps = attribs
            self.thisptr = new cmn.Genome(a_id, a_ni, a_nh, a_no, a_fs, a_oa, a_ha, a_st,
                                          deref((<Parameters>a_ps).thisptr))
    def __dealloc__(self):
        if not self.borrowed:
            del self.thisptr

    def __repr__(self):
        return 'ID {} Fitt {}'.format(self.thisptr.GetID(), self.thisptr.GetFitness())

    property Evaluated:
        def __get__(self): return self.thisptr.m_Evaluated

    def NumNeurons(self):
        return self.thisptr.NumNeurons()

    property Num_Neurons:
        def __get__(self): return self.thisptr.NumNeurons()
    
    def NumLinks(self):
        return self.thisptr.NumLinks()

    property Num_Links:
        def __get__(self): return self.thisptr.NumLinks()

    def NumInputs(self):
        return self.thisptr.NumInputs()

    property Num_Inputs:
        def __get__(self): return self.thisptr.NumInputs()
    
    def NumOutputs(self):
        return self.thisptr.NumOutputs()

    property Num_Outputs:
        def __get__(self): return self.thisptr.NumOutputs()
    
    def GetFitness(self):
        return self.thisptr.GetFitness()

    property Fitness:
        def __get__(self): return self.thisptr.GetFitness()
        def __set__(self, val): self.thisptr.SetFitness(val)
    
    def SetFitness(self, a_f):
        self.thisptr.SetFitness(a_f)
    
    def GetID(self):
        return self.thisptr.GetID()

    property ID:
        def __get__(self): return self.thisptr.GetID()
        def __set__(self, val): self.thisptr.SetID(val)
    
    def CalculateDepth(self):
        self.thisptr.CalculateDepth()
    
    def GetDepth(self):
        return self.thisptr.GetDepth()
    
    def BuildPhenotype(self, NeuralNetwork net):
        self.thisptr.BuildPhenotype(deref(net.thisptr))
    
    def BuildHyperNEATPhenotype(self, NeuralNetwork net, Substrate subst):
        self.thisptr.BuildHyperNEATPhenotype(deref(net.thisptr), deref(subst.thisptr))

    def BuildESHyperNEATPhenotype(Genome self, NeuralNetwork a_net, Substrate subst, Parameters params):
        self.thisptr.BuildESHyperNEATPhenotype(deref(a_net.thisptr), deref(subst.thisptr), deref(params.thisptr))
    
    def Save(self, str a_filename):
        self.thisptr.Save(a_filename)
    
    def IsEvaluated(self):
        return self.thisptr.IsEvaluated()
    
    def SetEvaluated(self):
        self.thisptr.SetEvaluated()
    
    def ResetEvaluated(self):
        self.thisptr.ResetEvaluated()


cdef class Species:
    cdef cmn.Species *thisptr      # hold a C++ instance which we're wrapping
    cdef bint borrowed
    def __cinit__(self, Genome a_Seed, int a_id):
        if a_Seed is None and a_id == -1:
            return
        self.thisptr = new cmn.Species(deref(a_Seed.thisptr), a_id)
        self.borrowed = False

    def __dealloc__(self):
        if not self.borrowed:
            del self.thisptr

    def __repr__(self):
        return 'ID {} AgeGens {}'.format(self.thisptr.ID(), self.thisptr.AgeGens())

    def GetBestFitness(self):
        return self.thisptr.GetBestFitness()

    def NumIndividuals(self):
        return self.thisptr.NumIndividuals()

    def GetLeader(self):
       return pyGenomeFromConstant(self.thisptr.GetLeader())

    property BestFitness:
        def __get__(self): return self.thisptr.m_BestFitness
        def __set__(self, m_BestFitness): self.thisptr.m_BestFitness = m_BestFitness

    property BestGenome:
       def __get__(self): return pyGenomeFromConstant(self.thisptr.m_BestGenome)
       def __set__(self,  Genome m_BestGenome): self.thisptr.m_BestGenome = deref(m_BestGenome.thisptr)

    property GensNoImprovement:
        def __get__(self): return self.thisptr.m_GensNoImprovement
        def __set__(self, m_GensNoImprovement): self.thisptr.m_GensNoImprovement = m_GensNoImprovement

    property B:
        def __get__(self): return self.thisptr.m_B
        def __set__(self, m_B): self.thisptr.m_B = m_B

    property Individuals:
        def __get__(self): return genomesVectorToList(self.thisptr.m_Individuals)

    property ID:
        def __get__(self): return self.thisptr.ID()

    property Age:
        def __get__(self): return self.thisptr.AgeGens()


cdef class Population:
    cdef cmn.Population *thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, *attribs):
        cdef Genome a_g
        cdef Parameters a_ps
        cdef int a_RNG_seed
        if len(attribs) == 1:
            self.thisptr = new cmn.Population(attribs[0])
        else:
            a_g, a_ps, a_r, a_rr, a_RNG_seed = attribs
            self.thisptr = new cmn.Population(deref((<Genome>a_g).thisptr), deref((<Parameters>a_ps).thisptr),
            a_r, a_rr, a_RNG_seed)

    def __dealloc__(self):
        del self.thisptr

    def GetSearchMode(self):
        return self.thisptr.GetSearchMode()

    def GetCurrentMPC(self):
        return self.thisptr.GetCurrentMPC()

    def GetBaseMPC(self):
        return self.thisptr.GetBaseMPC()

    def NumGenomes(self):
        return self.thisptr.NumGenomes()

    def GetGeneration(self):
        return self.thisptr.GetGeneration()

    def GetBestFitnessEver(self):
        return self.thisptr.GetBestFitnessEver()

    def GetBestGenome(self):
       return pyGenomeFromConstant(self.thisptr.GetBestGenome())

    def GetStagnation(self):
        return self.thisptr.GetStagnation()

    def GetMPCStagnation(self):
        return self.thisptr.GetMPCStagnation()

    def GetNextGenomeID(self):
        return self.thisptr.GetNextGenomeID()

    def GetNextSpeciesID(self):
        return self.thisptr.GetNextSpeciesID()

    def Epoch(self):
        return self.thisptr.Epoch()

    def Save(self, a_FileName):
        return self.thisptr.Save(a_FileName)

    #def Tick(self, a_deleted_genome):
    #    return self.thisptr.Tick

    property RNG:
        '''Random number generator'''
        def __get__(self): return buildRNG(self.thisptr.m_RNG)

    property Parameters:
        '''Evolution parameters'''
        def __get__(self): return buildParams(self.thisptr.m_Parameters)

    property Generation:
        '''Current generation'''
        def __get__(self): return self.thisptr.m_Generation

    property Species:
        '''The list of species'''
        def __get__(self):
            cdef list newspecies = speciesVectorToList(self.thisptr.m_Species)
            return newspecies

class NeuronType:
    NONE = cmn.NONE
    INPUT  = cmn.INPUT
    BIAS   = cmn.BIAS
    HIDDEN = cmn.HIDDEN
    OUTPUT = cmn.OUTPUT

class ActivationFunction:
    SIGNED_SIGMOID   = cmn.SIGNED_SIGMOID
    UNSIGNED_SIGMOID = cmn.UNSIGNED_SIGMOID
    TANH             = cmn.TANH
    TANH_CUBIC       = cmn.TANH_CUBIC
    SIGNED_STEP      = cmn.SIGNED_STEP
    UNSIGNED_STEP    = cmn.UNSIGNED_STEP
    SIGNED_GAUSS     = cmn.SIGNED_GAUSS
    UNSIGNED_GAUSS   = cmn.UNSIGNED_GAUSS
    ABS              = cmn.ABS
    SIGNED_SINE      = cmn.SIGNED_SINE
    UNSIGNED_SINE    = cmn.UNSIGNED_SINE
    # SIGNED_SQUARE    = cmn.SIGNED_SQUARE
    # UNSIGNED_SQUARE  = cmn.UNSIGNED_SQUARE
    LINEAR           = cmn.LINEAR

class SearchMode:
    COMPLEXIFYING = cmn.COMPLEXIFYING
    SIMPLIFYING   = cmn.SIMPLIFYING
    BLENDED       = cmn.BLENDED


cdef Genome pyGenomeFromReference(cmn.Genome& cval):
    cdef Genome ret = Genome()
    del ret.thisptr
    ret.thisptr = &cval
    ret.borrowed = True
    return ret

cdef Genome pyGenomeFromConstant(const cmn.Genome& cval):
    cdef Genome ret = Genome()
    ret.thisptr = new cmn.Genome(cval)
    return ret

cdef list genomesVectorToList(vector[cmn.Genome]& vec):
    cdef list ret = [None] * vec.size()
    cdef unsigned int i
    cdef cmn.Genome* p
    for i in range(vec.size()):
        p = &(<cmn.Genome&>vec.at(i))
        ret[i] = pyGenomeFromReference(deref(p))
    return ret

cdef Neuron pyNeuronFromReference(cmn.Neuron& cval):
    cdef Neuron ret = Neuron()
    del ret.thisptr
    ret.thisptr = &cval
    ret.borrowed = True
    return ret

cdef list neuronsVectorToList(const vector[cmn.Neuron]& vec):
    cdef list ret = [None] * vec.size()
    cdef unsigned int i
    cdef cmn.Neuron* np
    for i in range(vec.size()):
        np = &(<cmn.Neuron&>vec.at(i))
        ret[i] = pyNeuronFromReference(deref(np))
    return ret

cdef vector[cmn.Neuron] neuronsListToVector(list l):
    cdef unsigned int i, llen = len(l)
    cdef vector[cmn.Neuron] ret = vector[cmn.Neuron](llen)
    for i in range(llen):
        if not isinstance(l[i], Neuron):
            raise TypeError('expected \'Neuron\', got \'{}\''.format(type(l[i])))
        ret[i] = deref((<Neuron>(l[i])).thisptr)
        l[i].borrowed = True
    return ret

cdef Connection pyConnectionFromReference(cmn.Connection& cval):
    cdef Connection ret = Connection()
    del ret.thisptr
    ret.thisptr = &cval
    ret.borrowed = True
    return ret

cdef list connectionsVectorToList(vector[cmn.Connection]& vec):
    cdef list ret = [None] * vec.size()
    cdef unsigned int i
    cdef cmn.Connection* p
    for i in range(vec.size()):
        p = &(<cmn.Connection&>vec.at(i))
        ret[i] = pyConnectionFromReference(deref(p))
    return ret

cdef vector[cmn.Connection] connectionsListToVector(list l):
    cdef unsigned int i, llen = len(l)
    cdef vector[cmn.Connection] ret = vector[cmn.Connection](llen)
    for i in range(llen):
        if not isinstance(l[i], Neuron):
            raise TypeError('expected \'Connection\', got \'{}\''.format(type(l[i])))
        ret[i] = deref((<Connection>(l[i])).thisptr)
        l[i].borrowed = True
    return ret

cdef Species pySpeciesFromReference(cmn.Species& cval):
    cdef Species ret = Species(None, -1)
    ret.thisptr = &cval
    ret.borrowed = True
    return ret

cdef list speciesVectorToList(vector[cmn.Species]& vec):
    cdef list ret = [None] * vec.size()
    cdef unsigned int i
    cdef cmn.Species* p
    for i in range(vec.size()):
        p = &(<cmn.Species&>vec.at(i))
        ret[i] = pySpeciesFromReference(deref(p))
    return ret

cdef RNG buildRNG(cmn.RNG &rng):
    cdef RNG tRNG = RNG()
    del tRNG.thisptr
    tRNG.thisptr = &rng
    tRNG.borrowed = True
    return tRNG

cdef Parameters buildParams(cmn.Parameters& par):
    cdef Parameters tPar = Parameters()
    del tPar.thisptr
    tPar.thisptr = &par
    tPar.borrowed = True
    return tPar
