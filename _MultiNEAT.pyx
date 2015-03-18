# distutils: language = c++

from libcpp.vector cimport vector
from libcpp cimport bool
from cython.operator cimport dereference as deref, preincrement as preinc

"""
#############################################

Enums

#############################################
"""
cdef extern from "src/Genes.h" namespace "NEAT":
    cpdef enum NeuronType:
        NONE
        INPUT
        BIAS
        HIDDEN
        OUTPUT

    cpdef enum ActivationFunction:
        SIGNED_SIGMOID
        UNSIGNED_SIGMOID
        TANH
        TANH_CUBIC
        SIGNED_STEP
        UNSIGNED_STEP
        SIGNED_GAUSS
        UNSIGNED_GAUSS
        ABS
        SIGNED_SINE
        UNSIGNED_SINE
        SIGNED_SQUARE
        UNSIGNED_SQUARE
        LINEAR

cdef extern from "src/Population.h" namespace "NEAT":
    cpdef enum SearchMode:
        COMPLEXIFYING
        SIMPLIFYING
        BLENDED


"""
#############################################

RNG class

#############################################
"""

cdef extern from "src/Random.h" namespace "NEAT":
    cdef cppclass RNG:
        RNG() except +
        void Seed(int seed)
        void TimeSeed()
        int RandPosNeg()
        int RandInt(int x, int y)
        double RandFloat()
        double RandFloatClamped()
        double RandGaussClamped()
        int Roulette(vector[double]& a_probs)

cdef class pyRNG:
    cdef RNG *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new RNG()
    def __dealloc__(self):
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
        return self.thisptr.RandFloatClamped()
    def RandGaussClamped(self):
        return self.thisptr.RandGaussClamped()
    def Roulette(self, a_probs):
        return self.thisptr.Roulette(a_probs)

"""
#############################################

Parameters class

#############################################
"""

cdef extern from "src/Parameters.h" namespace "NEAT":
    cdef cppclass Parameters:
        unsigned int PopulationSize;
        bool DynamicCompatibility;
        unsigned int MinSpecies;
        unsigned int MaxSpecies;
        bool InnovationsForever;
        bool AllowClones;
        unsigned int YoungAgeTreshold;
        double YoungAgeFitnessBoost;
        unsigned int SpeciesMaxStagnation;
        double StagnationDelta;
        unsigned int OldAgeTreshold;
        double OldAgePenalty;
        bool DetectCompetetiveCoevolutionStagnation;
        int KillWorstSpeciesEach;
        int KillWorstAge;
        double SurvivalRate;
        double CrossoverRate;
        double OverallMutationRate;
        double InterspeciesCrossoverRate;
        double MultipointCrossoverRate;
        bool RouletteWheelSelection;
        bool PhasedSearching;
        bool DeltaCoding;
        unsigned int SimplifyingPhaseMPCTreshold;
        unsigned int SimplifyingPhaseStagnationTreshold;
        unsigned int ComplexityFloorGenerations;
        unsigned int NoveltySearch_K;
        double NoveltySearch_P_min;
        bool NoveltySearch_Dynamic_Pmin;
        unsigned int NoveltySearch_No_Archiving_Stagnation_Treshold;
        double NoveltySearch_Pmin_lowering_multiplier;
        double NoveltySearch_Pmin_min;
        unsigned int NoveltySearch_Quick_Archiving_Min_Evaluations;
        double NoveltySearch_Pmin_raising_multiplier;
        unsigned int NoveltySearch_Recompute_Sparseness_Each;
        double MutateAddNeuronProb;
        bool SplitRecurrent;
        bool SplitLoopedRecurrent;
        int NeuronTries;
        double MutateAddLinkProb;
        double MutateAddLinkFromBiasProb;
        double MutateRemLinkProb;
        double MutateRemSimpleNeuronProb;
        unsigned int LinkTries;
        double RecurrentProb;
        double RecurrentLoopProb;
        double MutateWeightsProb;
        double MutateWeightsSevereProb;
        double WeightMutationRate;
        double WeightMutationMaxPower;
        double WeightReplacementMaxPower;
        double MaxWeight;
        double MutateActivationAProb;
        double MutateActivationBProb;
        double ActivationAMutationMaxPower;
        double ActivationBMutationMaxPower;
        double TimeConstantMutationMaxPower;
        double BiasMutationMaxPower;
        double MinActivationA;
        double MaxActivationA;
        double MinActivationB;
        double MaxActivationB;
        double MutateNeuronActivationTypeProb;
        double ActivationFunction_SignedSigmoid_Prob;
        double ActivationFunction_UnsignedSigmoid_Prob;
        double ActivationFunction_Tanh_Prob;
        double ActivationFunction_TanhCubic_Prob;
        double ActivationFunction_SignedStep_Prob;
        double ActivationFunction_UnsignedStep_Prob;
        double ActivationFunction_SignedGauss_Prob;
        double ActivationFunction_UnsignedGauss_Prob;
        double ActivationFunction_Abs_Prob;
        double ActivationFunction_SignedSine_Prob;
        double ActivationFunction_UnsignedSine_Prob;
        double ActivationFunction_SignedSquare_Prob;
        double ActivationFunction_UnsignedSquare_Prob;
        double ActivationFunction_Linear_Prob;
        double MutateNeuronTimeConstantsProb;
        double MutateNeuronBiasesProb;
        double MinNeuronTimeConstant;
        double MaxNeuronTimeConstant;
        double MinNeuronBias;
        double MaxNeuronBias;
        double DisjointCoeff;
        double ExcessCoeff;
        double ActivationADiffCoeff;
        double ActivationBDiffCoeff;
        double WeightDiffCoeff;
        double TimeConstantDiffCoeff;
        double BiasDiffCoeff;
        double ActivationFunctionDiffCoeff;
        double CompatTreshold;
        double MinCompatTreshold;
        double CompatTresholdModifier;
        unsigned int CompatTreshChangeInterval_Generations;
        unsigned int CompatTreshChangeInterval_Evaluations;

        Parameters() except +

        int Load(const char* filename);
        void Save(const char* filename);
        void Reset();


cdef class pyParameters:
    cdef Parameters *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new Parameters()
    def __dealloc__(self):
        del self.thisptr

    def Load(self, filename):
        self.thisptr.Load(filename)
    def Save(self, filename):
        self.thisptr.Load(filename)
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
        
    property ActivationFunction_SignedSquare_Prob:
        def __get__(self): return self.thisptr.ActivationFunction_SignedSquare_Prob
        def __set__(self, ActivationFunction_SignedSquare_Prob): self.thisptr.ActivationFunction_SignedSquare_Prob = ActivationFunction_SignedSquare_Prob
        
    property ActivationFunction_UnsignedSquare_Prob:
        def __get__(self): return self.thisptr.ActivationFunction_UnsignedSquare_Prob
        def __set__(self, ActivationFunction_UnsignedSquare_Prob): self.thisptr.ActivationFunction_UnsignedSquare_Prob = ActivationFunction_UnsignedSquare_Prob
        
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

"""
#############################################

NeuralNetwork class

#############################################
"""

cdef extern from "src/NeuralNetwork.h" namespace "NEAT":
    cdef cppclass Connection:
        unsigned short int m_source_neuron_idx;
        unsigned short int m_target_neuron_idx;
        double m_weight;
        bool m_recur_flag;

    cdef cppclass Neuron:
        double m_activation;
        ActivationFunction m_activation_function_type;
        double m_a;
        double m_b;
        double m_timeconst;
        double m_bias;
        double m_x;
        double m_y;
        double m_z;
        double m_sx;
        double m_sy;
        double m_sz;
        vector[double] m_substrate_coords;
        double m_split_y;
        NeuronType m_type;

    cdef cppclass NeuralNetwork:
        unsigned short m_num_inputs;
        unsigned short m_num_outputs;
        vector[Neuron] m_neurons;
        vector[Connection] m_connections;

        NeuralNetwork() except +
        NeuralNetwork(bool x) except +

        void InitRTRLMatrix();
        void ActivateFast();
        void Activate();
        void ActivateUseInternalBias();
        void ActivateLeaky(double step);

        void RTRL_update_gradients();
        void RTRL_update_error(double a_target);
        void RTRL_update_weights();
        void Adapt(Parameters& a_Parameters);

        void Flush();
        void FlushCube();
        void Input(vector[double]& a_Inputs);
        vector[double] Output();


cdef class pyConnection:
    cdef Connection *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new Connection()
    def __dealloc__(self):
        del self.thisptr

    property m_source_neuron_idx:
            def __get__(self): return self.thisptr.m_source_neuron_idx
            def __set__(self, m_source_neuron_idx): self.thisptr.m_source_neuron_idx = m_source_neuron_idx
            
    property m_target_neuron_idx:
            def __get__(self): return self.thisptr.m_target_neuron_idx
            def __set__(self, m_target_neuron_idx): self.thisptr.m_target_neuron_idx = m_target_neuron_idx
            
    property m_weight:
            def __get__(self): return self.thisptr.m_weight
            def __set__(self, m_weight): self.thisptr.m_weight = m_weight
            
    property m_recur_flag:
            def __get__(self): return self.thisptr.m_recur_flag
            def __set__(self, m_recur_flag): self.thisptr.m_recur_flag = m_recur_flag
            
            
cdef class pyNeuron:
    cdef Neuron *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new Neuron()
    def __dealloc__(self):
        del self.thisptr

    property m_activation:
            def __get__(self): return self.thisptr.m_activation
            def __set__(self, m_activation): self.thisptr.m_activation = m_activation
            
    property m_activation_function_type:
            def __get__(self): return self.thisptr.m_activation_function_type
            def __set__(self, m_activation_function_type): self.thisptr.m_activation_function_type = m_activation_function_type
            
    property m_a:
            def __get__(self): return self.thisptr.m_a
            def __set__(self, m_a): self.thisptr.m_a = m_a
            
    property m_b:
            def __get__(self): return self.thisptr.m_b
            def __set__(self, m_b): self.thisptr.m_b = m_b
            
    property m_timeconst:
            def __get__(self): return self.thisptr.m_timeconst
            def __set__(self, m_timeconst): self.thisptr.m_timeconst = m_timeconst
            
    property m_bias:
            def __get__(self): return self.thisptr.m_bias
            def __set__(self, m_bias): self.thisptr.m_bias = m_bias
            
    property m_x:
            def __get__(self): return self.thisptr.m_x
            def __set__(self, m_x): self.thisptr.m_x = m_x
            
    property m_y:
            def __get__(self): return self.thisptr.m_y
            def __set__(self, m_y): self.thisptr.m_y = m_y
            
    property m_z:
            def __get__(self): return self.thisptr.m_z
            def __set__(self, m_z): self.thisptr.m_z = m_z
            
    property m_sx:
            def __get__(self): return self.thisptr.m_sx
            def __set__(self, m_sx): self.thisptr.m_sx = m_sx
            
    property m_sy:
            def __get__(self): return self.thisptr.m_sy
            def __set__(self, m_sy): self.thisptr.m_sy = m_sy
            
    property m_sz:
            def __get__(self): return self.thisptr.m_sz
            def __set__(self, m_sz): self.thisptr.m_sz = m_sz
            
    property m_substrate_coords:
            def __get__(self): return self.thisptr.m_substrate_coords
            def __set__(self, m_substrate_coords): self.thisptr.m_substrate_coords = m_substrate_coords
            
    property m_split_y:
            def __get__(self): return self.thisptr.m_split_y
            def __set__(self, m_split_y): self.thisptr.m_split_y = m_split_y
            
    property m_type:
            def __get__(self): return self.thisptr.m_type
            def __set__(self, m_type): self.thisptr.m_type = m_type
        

            
            
cdef class pyNeuralNetwork:
    cdef NeuralNetwork *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new NeuralNetwork()
    def __cinit__(self, x):
        self.thisptr = new NeuralNetwork(x)
    def __dealloc__(self):
        del self.thisptr

    def InitRTRLMatrix(self):
        return self.thisptr.InitRTRLMatrix()
    
    def ActivateFast(self):
        return self.thisptr.ActivateFast()
    
    def Activate(self):
        return self.thisptr.Activate()
    
    def ActivateUseInternalBias(self):
        return self.thisptr.ActivateUseInternalBias()
    
    def ActivateLeaky(self, step):
        return self.thisptr.ActivateLeaky(step)
    
    def RTRL_update_gradients(self):
        return self.thisptr.RTRL_update_gradients()
    
    def RTRL_update_error(self, double a_target):
        return self.thisptr.RTRL_update_error(a_target)
    
    def RTRL_update_weights(self):
        return self.thisptr.RTRL_update_weights()
    
    def Adapt(self, pyParameters a_Parameters):
        return self.thisptr.Adapt(deref(a_Parameters.thisptr))
    
    def Flush(self):
        return self.thisptr.Flush()
    
    def FlushCube(self):
        return self.thisptr.FlushCube()
    
    def Input(self, a_Inputs):
        return self.thisptr.Input(a_Inputs)
    
    def Output(self):
        return self.thisptr.Output()


    property m_num_inputs:
            def __get__(self): return self.thisptr.m_num_inputs
            def __set__(self, m_num_inputs): self.thisptr.m_num_inputs = m_num_inputs
            
    property m_num_outputs:
            def __get__(self): return self.thisptr.m_num_outputs
            def __set__(self, m_num_outputs): self.thisptr.m_num_outputs = m_num_outputs
            
    #property m_neurons:
    #        def __get__(self): return self.thisptr.m_neurons
    #        def __set__(self, vector[Neuron] m_neurons): self.thisptr.m_neurons = m_neurons
            
    #property m_connections:
    #        def __get__(self): return self.thisptr.m_connections
    #        def __set__(self, vector[Connection] m_connections): self.thisptr.m_connections = m_connections
"""
#############################################

Substrate class

#############################################
"""
#cdef vector[ vector[double] ] x
#cdef vector[double] y
#y.push_back(1)
#x.push_back(y)

cdef extern from "src/Substrate.h" namespace "NEAT":
    cdef cppclass Substrate:
        vector[vector[double]] m_input_coords;
        vector[vector[double]] m_hidden_coords;
        vector[vector[double]] m_output_coords;

        bool m_leaky;
        bool m_with_distance;

        bool m_allow_input_hidden_links;
        bool m_allow_input_output_links;
        bool m_allow_hidden_hidden_links;
        bool m_allow_hidden_output_links;
        bool m_allow_output_hidden_links;
        bool m_allow_output_output_links;
        bool m_allow_looped_hidden_links;
        bool m_allow_looped_output_links;

        ActivationFunction m_hidden_nodes_activation;
        ActivationFunction m_output_nodes_activation;

        double m_link_threshold;
        double m_max_weight_and_bias;
        double m_min_time_const;
        double m_max_time_const;

        Substrate();
        Substrate(vector[vector[double]]& a_inputs,
                  vector[vector[double]]& a_hidden,
                  vector[vector[double]]& a_outputs);

        int GetMinCPPNInputs();
        int GetMinCPPNOutputs();
        void PrintInfo();
        

cdef class pySubstrate:
    cdef Substrate *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new Substrate()
    def __cinit__(self, i, h, o):
        self.thisptr = new Substrate(i, h, o)
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
            
    property m_link_threshold:
            def __get__(self): return self.thisptr.m_link_threshold
            def __set__(self, m_link_threshold): self.thisptr.m_link_threshold = m_link_threshold
            
    property m_max_weight_and_bias:
            def __get__(self): return self.thisptr.m_max_weight_and_bias
            def __set__(self, m_max_weight_and_bias): self.thisptr.m_max_weight_and_bias = m_max_weight_and_bias
            
    property m_min_time_const:
            def __get__(self): return self.thisptr.m_min_time_const
            def __set__(self, m_min_time_const): self.thisptr.m_min_time_const = m_min_time_const
            
    property m_max_time_const:
            def __get__(self): return self.thisptr.m_max_time_const
            def __set__(self, m_max_time_const): self.thisptr.m_max_time_const = m_max_time_const
        

"""
#############################################

Genome class

#############################################
"""

cdef extern from "src/Genome.h" namespace "NEAT":
    cdef cppclass Genome:
        Genome() except +
        Genome(const char* a_filename);
        Genome(unsigned int a_ID,
               unsigned int a_NumInputs,
               unsigned int a_NumHidden,
               unsigned int a_NumOutputs,
               bool a_FS_NEAT, ActivationFunction a_OutputActType,
               ActivationFunction a_HiddenActType,
               unsigned int a_SeedType,
               const Parameters& a_Parameters);

        unsigned int NumNeurons();
        unsigned int NumLinks();
        unsigned int NumInputs();
        unsigned int NumOutputs();

        double GetFitness();
        void SetFitness(double a_f);
        unsigned int GetID();
        void CalculateDepth();
        unsigned int GetDepth();

        void BuildPhenotype(NeuralNetwork& net);
        void BuildHyperNEATPhenotype(NeuralNetwork& net, Substrate& subst);

        void Save(const char* a_filename);

        bool IsEvaluated();
        void SetEvaluated();
        void ResetEvaluated();


cdef class pyGenome:
    cdef Genome *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new Genome()
    def __cinit__(self, fname):
        self.thisptr = new Genome(fname)
    def __cinit__(self, a_id, a_ni, a_nh, a_no, a_fs, a_oa, a_ha, a_st, pyParameters a_ps):
        self.thisptr = new Genome(a_id, a_ni, a_nh, a_no, a_fs, a_oa, a_ha, a_st, deref(a_ps.thisptr))
    def __dealloc__(self):
        del self.thisptr
        
    def NumNeurons(self):
        return self.thisptr.NumNeurons()
    
    def NumLinks(self):
        return self.thisptr.NumLinks()

    def NumInputs(self):
        return self.thisptr.NumInputs()
    
    def NumOutputs(self):
        return self.thisptr.NumOutputs()
    
    def GetFitness(self):
        return self.thisptr.GetFitness()
    
    def SetFitness(self, a_f):
        return self.thisptr.SetFitness(a_f)
    
    def GetID(self):
        return self.thisptr.GetID()
    
    def CalculateDepth(self):
        return self.thisptr.CalculateDepth()
    
    def GetDepth(self):
        return self.thisptr.GetDepth()
    
    def BuildPhenotype(self, pyNeuralNetwork net):
        return self.thisptr.BuildPhenotype(deref(net.thisptr))
    
    def BuildHyperNEATPhenotype(self, pyNeuralNetwork net, pySubstrate subst):
        return self.thisptr.BuildHyperNEATPhenotype(deref(net.thisptr), deref(subst.thisptr))
    
    def Save(self, a_filename):
        return self.thisptr.Save(a_filename)
    
    def IsEvaluated(self):
        return self.thisptr.IsEvaluated()
    
    def SetEvaluated(self):
        return self.thisptr.SetEvaluated()
    
    def ResetEvaluated(self):
        return self.thisptr.ResetEvaluated()

"""
#############################################

Species class

#############################################
"""

"""
cdef extern from "src/Species.h" namespace "NEAT":
    cdef cppclass Species:
        Species() except +
        
        double m_BestFitness;
        Genome m_BestGenome;
        unsigned int m_GensNoImprovement;
        int m_R, m_G, m_B;
        vector[Genome] m_Individuals;
        
        double GetBestFitness();
        unsigned int NumIndividuals();
        Genome GetLeader();
        


cdef class pySpecies:
    cdef Species *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new Species()
    def __dealloc__(self):
        del self.thisptr

    def GetBestFitness(self):
        return self.thisptr.GetBestFitness()
    
    def NumIndividuals(self):
        return self.thisptr.NumIndividuals()
    
    #def GetLeader(self):
    #    return self.thisptr.GetLeader()

    property m_BestFitness:
        def __get__(self): return self.thisptr.m_BestFitness
        def __set__(self, m_BestFitness): self.thisptr.m_BestFitness = m_BestFitness
            
    #property m_BestGenome:
    #    def __get__(self): return self.thisptr.m_BestGenome
    #    def __set__(self, m_BestGenome): self.thisptr.m_BestGenome = m_BestGenome
            
    property m_GensNoImprovement:
        def __get__(self): return self.thisptr.m_GensNoImprovement
        def __set__(self, m_GensNoImprovement): self.thisptr.m_GensNoImprovement = m_GensNoImprovement
            
    property m_B:
        def __get__(self): return self.thisptr.m_B
        def __set__(self, m_B): self.thisptr.m_B = m_B
"""

        
"""
#############################################

Population class

#############################################
"""

cdef extern from "src/Population.h" namespace "NEAT":
    cdef cppclass Population:
        Population(const Genome& a_G, const Parameters& a_Parameters, bool a_RandomizeWeights, double a_RandomRange);
        Population(const char* a_FileName);

        RNG m_RNG;
        Parameters m_Parameters;
        unsigned int m_Generation;
        unsigned int m_NumEvaluations;
        #vector[Species] m_Species;

        SearchMode GetSearchMode(); 
        double GetCurrentMPC(); 
        double GetBaseMPC(); 
    
        unsigned int NumGenomes(); 
    
        unsigned int GetGeneration();
        double GetBestFitnessEver();
        Genome GetBestGenome();

        unsigned int GetStagnation();
        unsigned int GetMPCStagnation();
    
        unsigned int GetNextGenomeID();
        unsigned int GetNextSpeciesID();
        
        void Epoch();
    
        void Save(const char* a_FileName);
        Genome* Tick(Genome& a_deleted_genome);


cdef class pyPopulation:
    cdef Population *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, pyGenome a_g, pyParameters a_ps, a_r, a_rr):
        self.thisptr = new Population(deref(a_g.thisptr), deref(a_ps.thisptr), a_r, a_rr)
    def __cinit__(self, fn):
        self.thisptr = new Population(fn)
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
    
    #def GetBestGenome(self):
    #    return self.thisptr.GetBestGenome()
    
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
    #    return self.thisptr.Tick(a_deleted_genome)
    
    