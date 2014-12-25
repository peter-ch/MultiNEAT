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





"""
#############################################

NeuralNetwork class

#############################################
"""

cdef extern from "src/NeuralNetwork.h" namespace "NEAT":
    cdef cppclass Connection:
        unsigned short int m_source_neuron_idx, m_target_neuron_idx
        double m_weight
        bool m_recur_flag

    cdef cppclass Neuron:
        double m_activation
        ActivationFunction m_activation_function_type
        double m_a, m_b, m_timeconst, m_bias
        double m_x, m_y, m_z
        double m_sx, m_sy, m_sz
        vector[double] m_substrate_coords
        double m_split_y
        NeuronType m_type

    cdef cppclass NeuralNetwork:
        unsigned short m_num_inputs, m_num_outputs
        vector[Neuron] m_neurons
        vector[Connection] m_connections

        NeuralNetwork() except +
        NeuralNetwork(bool) except +

        void InitRTRLMatrix()
        void ActivateFast()
        void Activate()
        void ActivateUseInternalBias()
        void ActivateLeaky(double step)

        void RTRL_update_gradients()
        void RTRL_update_error(double a_target)
        void RTRL_update_weights()
        void Adapt(Parameters& a_Parameters)

        void Flush()
        void FlushCube()
        void Input(vector[double]& a_Inputs)
        vector[double] Output()



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

        Substrate()
        Substrate(vector[vector[double]]& a_inputs,
                  vector[vector[double]]& a_hidden,
                  vector[vector[double]]& a_outputs)

        int GetMinCPPNInputs()
        int GetMinCPPNOutputs()
        void PrintInfo()


"""
#############################################

Genome class

#############################################
"""

cdef extern from "src/Genome.h" namespace "NEAT":
    cdef cppclass Genome:
        Genome() except +
        Genome(const char* a_filename)
        Genome(unsigned int a_ID,
               unsigned int a_NumInputs,
               unsigned int a_NumHidden,
               unsigned int a_NumOutputs,
               bool a_FS_NEAT, ActivationFunction a_OutputActType,
               ActivationFunction a_HiddenActType,
               unsigned int a_SeedType,
               const Parameters& a_Parameters);

        unsigned int NumNeurons()
        unsigned int NumLinks()
        unsigned int NumInputs()
        unsigned int NumOutputs()

        double GetFitness()
        void SetFitness(double a_f)
        unsigned int GetID()
        void CalculateDepth()
        unsigned int GetDepth()

        void BuildPhenotype(NeuralNetwork& net)
        void BuildHyperNEATPhenotype(NeuralNetwork& net, Substrate& subst)

        void Save(const char* a_filename);

        bool IsEvaluated()
        void SetEvaluated()
        void ResetEvaluated()


