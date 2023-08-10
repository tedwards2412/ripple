
 typedef struct tagIMRPhenomDAmplitudeCoefficients {
   double eta;         // symmetric mass-ratio
   double etaInv;      // 1/eta
   double chi12;       // chi1*chi1;
   double chi22;       // chi2*chi2;
   double eta2;        // eta*eta;
   double eta3;        // eta*eta*eta;
   double Seta;        // sqrt(1.0 - 4.0*eta);
   double SetaPlus1;   // (1.0 + Seta);
   double chi1, chi2;  // dimensionless aligned spins, convention m1 >= m2.
   double q;           // asymmetric mass-ratio (q>=1)
   double chi;         // PN reduced spin parameter
   double fRD;         // ringdown frequency
   double fDM;         // imaginary part of the ringdown frequency (damping time)
  
   double fmaxCalc;    // frequency at which the mrerger-ringdown amplitude is maximum
  
   // Phenomenological inspiral amplitude coefficients
   double rho1;
   double rho2;
   double rho3;
  
   // Phenomenological intermediate amplitude coefficients
   double delta0;
   double delta1;
   double delta2;
   double delta3;
   double delta4;
  
   // Phenomenological merger-ringdown amplitude coefficients
   double gamma1;
   double gamma2;
   double gamma3;
  
   // Coefficients for collocation method. Used in intermediate amplitude model
   double f1, f2, f3;
   double v1, v2, v3;
   double d1, d2;
  
   // Transition frequencies for amplitude
   // We don't *have* to store them, but it may be clearer.
   double fInsJoin;    // Ins = Inspiral
   double fMRDJoin;    // MRD = Merger-Ringdown
 }
 IMRPhenomDAmplitudeCoefficients;

   
 /**
   * Structure holding all coefficients for the phase
   */
 typedef struct tagIMRPhenomDPhaseCoefficients {
   double eta;         // symmetric mass-ratio
   double etaInv;      // 1/eta
   double eta2;        // eta*eta
   double Seta;        // sqrt(1.0 - 4.0*eta);
   double chi1, chi2;  // dimensionless aligned spins, convention m1 >= m2.
   double q;           // asymmetric mass-ratio (q>=1)
   double chi;         // PN reduced spin parameter
   double fRD;         // ringdown frequency
   double fDM;         // imaginary part of the ringdown frequency (damping time)
  
   // Phenomenological inspiral phase coefficients
   double sigma1;
   double sigma2;
   double sigma3;
   double sigma4;
   double sigma5;
  
   // Phenomenological intermediate phase coefficients
   double beta1;
   double beta2;
   double beta3;
  
   // Phenomenological merger-ringdown phase coefficients
   double alpha1;
   double alpha2;
   double alpha3;
   double alpha4;
   double alpha5;
  
   // C1 phase connection coefficients
   double C1Int;
   double C2Int;
   double C1MRD;
   double C2MRD;
  
   // Transition frequencies for phase
   double fInsJoin;    // Ins = Inspiral
   double fMRDJoin;    // MRD = Merger-Ringdown
 }
 IMRPhenomDPhaseCoefficients;