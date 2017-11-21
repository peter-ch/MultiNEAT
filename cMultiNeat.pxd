from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdio cimport FILE

"""
#############################################

Enums

#############################################
"""
cdef extern from "src/Genes.h" namespace "NEAT":
    cdef enum NeuronType:
        NONE
        INPUT
        BIAS
        HIDDEN
        OUTPUT

    cdef enum ActivationFunction:
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
        #SIGNED_SQUARE
        #UNSIGNED_SQUARE
        LINEAR

cdef extern from "src/Population.h" namespace "NEAT":
    cdef enum SearchMode:
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
        double RandFloatSigned()
        double RandGaussSigned()
        int Roulette(vector[double]& a_probs)


"""
#############################################

Parameters class

#############################################
"""

cdef extern from "src/Parameters.h" namespace "NEAT":
    cdef cppclass Parameters:
        unsigned int PopulationSize
        bool DynamicCompatibility
        unsigned int MinSpecies
        unsigned int MaxSpecies
        bool InnovationsForever
        bool AllowClones
        unsigned int YoungAgeTreshold
        double YoungAgeFitnessBoost
        unsigned int SpeciesMaxStagnation
        double StagnationDelta
        unsigned int OldAgeTreshold
        double OldAgePenalty
        bool DetectCompetetiveCoevolutionStagnation
        int KillWorstSpeciesEach
        int KillWorstAge
        double SurvivalRate
        double CrossoverRate
        double OverallMutationRate
        double InterspeciesCrossoverRate
        double MultipointCrossoverRate
        bool RouletteWheelSelection
        bool PhasedSearching
        bool DeltaCoding
        unsigned int SimplifyingPhaseMPCTreshold
        unsigned int SimplifyingPhaseStagnationTreshold
        unsigned int ComplexityFloorGenerations
        unsigned int NoveltySearch_K
        double NoveltySearch_P_min
        bool NoveltySearch_Dynamic_Pmin
        unsigned int NoveltySearch_No_Archiving_Stagnation_Treshold
        double NoveltySearch_Pmin_lowering_multiplier
        double NoveltySearch_Pmin_min
        unsigned int NoveltySearch_Quick_Archiving_Min_Evaluations
        double NoveltySearch_Pmin_raising_multiplier
        unsigned int NoveltySearch_Recompute_Sparseness_Each
        double MutateAddNeuronProb
        bool SplitRecurrent
        bool SplitLoopedRecurrent
        int NeuronTries
        double MutateAddLinkProb
        double MutateAddLinkFromBiasProb
        double MutateRemLinkProb
        double MutateRemSimpleNeuronProb
        unsigned int LinkTries
        double RecurrentProb
        double RecurrentLoopProb
        double MutateWeightsProb
        double MutateWeightsSevereProb
        double WeightMutationRate
        double WeightMutationMaxPower
        double WeightReplacementMaxPower
        double MaxWeight
        double MutateActivationAProb
        double MutateActivationBProb
        double ActivationAMutationMaxPower
        double ActivationBMutationMaxPower
        double TimeConstantMutationMaxPower
        double BiasMutationMaxPower
        double MinActivationA
        double MaxActivationA
        double MinActivationB
        double MaxActivationB
        double MutateNeuronActivationTypeProb
        double ActivationFunction_SignedSigmoid_Prob
        double ActivationFunction_UnsignedSigmoid_Prob
        double ActivationFunction_Tanh_Prob
        double ActivationFunction_TanhCubic_Prob
        double ActivationFunction_SignedStep_Prob
        double ActivationFunction_UnsignedStep_Prob
        double ActivationFunction_SignedGauss_Prob
        double ActivationFunction_UnsignedGauss_Prob
        double ActivationFunction_Abs_Prob
        double ActivationFunction_SignedSine_Prob
        double ActivationFunction_UnsignedSine_Prob
        #double ActivationFunction_SignedSquare_Prob
        #double ActivationFunction_UnsignedSquare_Prob
        double ActivationFunction_Linear_Prob
        double MutateNeuronTimeConstantsProb
        double MutateNeuronBiasesProb
        double MinNeuronTimeConstant
        double MaxNeuronTimeConstant
        double MinNeuronBias
        double MaxNeuronBias
        bool DontUseBiasNeuron
        bool AllowLoops
        double DisjointCoeff
        double ExcessCoeff
        double ActivationADiffCoeff
        double ActivationBDiffCoeff
        double WeightDiffCoeff
        double TimeConstantDiffCoeff
        double BiasDiffCoeff
        double ActivationFunctionDiffCoeff
        double CompatTreshold
        double MinCompatTreshold
        double CompatTresholdModifier
        unsigned int CompatTreshChangeInterval_Generations
        unsigned int CompatTreshChangeInterval_Evaluations

        # Fraction of individuals to be copied unchanged
        double Elitism 'EliteFraction'

        Parameters() except +

        int Load(const char* filename)
        void Save(const char* filename)
        void Reset()

        ##############
        # ES HyperNEAT params
        ##############

        double DivisionThreshold

        double VarianceThreshold

        # Used for Band prunning.
        double BandThreshold

        # Max and Min Depths of the quadtree
        unsigned int InitialDepth

        unsigned int MaxDepth

        # How many hidden layers before connecting nodes to output. At 0 there is
        # one hidden layer. At 1, there are two and so on.
        unsigned int IterationLevel

        # The Bias value for the CPPN queries.
        double CPPN_Bias

        # Quadtree Dimensions
        # The range of the tree. Typically set to 2,
        double Width
        double Height

        # The (x, y) coordinates of the tree
        double Qtree_X

        double Qtree_Y

        # Use Link Expression output
        bool Leo

        # Threshold above which a connection is expressed
        double LeoThreshold

        # Use geometric seeding. Currently only along the X axis. 1
        bool LeoSeed
        bool GeometrySeed


"""
#############################################

NeuralNetwork class

#############################################
"""

cdef extern from "src/NeuralNetwork.h" namespace "NEAT":
    cdef cppclass Connection:
        unsigned short int m_source_neuron_idx
        unsigned short int m_target_neuron_idx
        double m_weight
        bool m_recur_flag

    cdef cppclass Neuron:
        double m_activation
        ActivationFunction m_activation_function_type
        double m_a
        double m_b
        double m_timeconst
        double m_bias
        double m_x
        double m_y
        double m_z
        double m_sx
        double m_sy
        double m_sz
        vector[double] m_substrate_coords
        double m_split_y
        NeuronType m_type

    cdef cppclass NeuralNetwork:
        unsigned short m_num_inputs
        unsigned short m_num_outputs
        vector[Neuron] m_neurons
        vector[Connection] m_connections

        NeuralNetwork() except +
        NeuralNetwork(bool x) except +

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
        void Clear()

        # one-shot save/load
        void Save(const char* a_filename)
        bool Load(const char* a_filename)
        # save/load from already opened files for reading/writing
        void Save(FILE* a_file)

        void AddNeuron(const Neuron& a_n)
        void AddConnection(const Connection& a_c)
        Connection GetConnectionByIndex(unsigned int a_idx) const
        Neuron GetNeuronByIndex(unsigned int a_idx) const
        void SetInputOutputDimentions(const unsigned short a_i, const unsigned short a_o)
        unsigned int NumInputs() const
        unsigned int NumOutputs() const


cdef extern from "src/Substrate.h" namespace "NEAT":
    cdef cppclass Substrate:
        vector[vector[double]] m_input_coords
        vector[vector[double]] m_hidden_coords
        vector[vector[double]] m_output_coords

        bool m_leaky
        bool m_with_distance

        bool m_allow_input_hidden_links
        bool m_allow_input_output_links
        bool m_allow_hidden_hidden_links
        bool m_allow_hidden_output_links
        bool m_allow_output_hidden_links
        bool m_allow_output_output_links
        bool m_allow_looped_hidden_links
        bool m_allow_looped_output_links

        ActivationFunction m_hidden_nodes_activation
        ActivationFunction m_output_nodes_activation

        #double m_link_threshold
        double m_max_weight_and_bias
        double m_min_time_const
        double m_max_time_const

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
        Genome(const Genome& a_g) except +
        Genome(const char* a_filename)
        Genome(unsigned int a_ID,
               unsigned int a_NumInputs,
               unsigned int a_NumHidden,
               unsigned int a_NumOutputs,
               bool a_FS_NEAT, ActivationFunction a_OutputActType,
               ActivationFunction a_HiddenActType,
               unsigned int a_SeedType,
               const Parameters& a_Parameters)

        unsigned int NumNeurons()
        unsigned int NumLinks()
        unsigned int NumInputs()
        unsigned int NumOutputs()

        double GetFitness()
        void SetFitness(double a_f)
        unsigned int GetID()
        void SetID(int a_id)
        void CalculateDepth()
        unsigned int GetDepth()

        void BuildPhenotype(NeuralNetwork& net)
        void BuildHyperNEATPhenotype(NeuralNetwork& net, Substrate& subst)
        void BuildESHyperNEATPhenotype(NeuralNetwork& a_net, Substrate& subst, Parameters& params)

        void Save(const char* a_filename)
        # Saves this genome to an already opened file for writing
        void Save(FILE* a_fstream);

        bool m_Evaluated
        bool IsEvaluated()
        void SetEvaluated()
        void ResetEvaluated()


"""
#############################################

Species class

#############################################
"""

cdef extern from "src/Species.h" namespace "NEAT":
    cdef cppclass Species:
        Species(const Genome& a_Seed, int a_id) except +
        Species& copy 'operator='(const Species& a_g)

        double m_BestFitness
        Genome m_BestGenome
        unsigned int m_GensNoImprovement
        int m_R, m_G, m_B
        vector[Genome] m_Individuals

        # Access
        double GetBestFitness()
        void SetBestSpecies(bool t)
        void SetWorstSpecies(bool t)
        void IncreaseAgeGens()
        void ResetAgeGens()
        void IncreaseGensNoImprovement()
        void SetOffspringRqd(double a_ofs)
        double GetOffspringRqd()
        unsigned int NumIndividuals()
        void ClearIndividuals()
        int ID()
        int GensNoImprovement()
        int AgeGens()
        Genome GetIndividualByIdx(int a_idx)
        bool IsBestSpecies()
        bool IsWorstSpecies()
        void SetRepresentative(Genome& a_G)

        Genome GetLeader()


"""
#############################################

Population class

#############################################
"""

cdef extern from "src/Population.h" namespace "NEAT":
    cdef cppclass Population:
        Population(const Genome& a_G, const Parameters& a_Parameters, bool a_RandomizeWeights, double a_RandomRange,
        int a_RNG_seed)
        Population(const char* a_FileName)

        RNG m_RNG
        Parameters m_Parameters
        unsigned int m_Generation
        unsigned int m_NumEvaluations
        vector[Species] m_Species

        SearchMode GetSearchMode()
        double GetCurrentMPC()
        double GetBaseMPC()

        unsigned int NumGenomes()

        unsigned int GetGeneration()
        double GetBestFitnessEver()
        Genome GetBestGenome()

        unsigned int GetStagnation()
        unsigned int GetMPCStagnation()

        unsigned int GetNextGenomeID()
        unsigned int GetNextSpeciesID()

        void Epoch()

        void Save(const char* a_FileName)
        Genome* Tick(Genome& a_deleted_genome)
