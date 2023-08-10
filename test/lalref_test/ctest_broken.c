#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <complex.h>
#include <tgmath.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_math.h>
#include "ctest_phenomD.c"
//#include <gsl/gsl_sf_trig.h>
//#include <iostream>
//#include <lal/LALDatatypes.h>

#define ROTATEY(angle, vx, vy, vz)\
tmp1 = vx*cos(angle) + vz*sin(angle);\
tmp2 = - vx*sin(angle) + vz*cos(angle);\
vx = tmp1;\
vz = tmp2

#define ROTATEZ(angle, vx, vy, vz)\
tmp1 = vx*cos(angle) - vy*sin(angle);\
tmp2 = vx*sin(angle) + vy*cos(angle);\
vx = tmp1;\
vy = tmp2


#define REAL8 double
#define COMPLEX16 double complex
#define MAX_TOL_ATAN 1e-10
#define XLAL_EFAULT 0
#define XLAL_EDOM 0
#define LAL_MSUN_SI  1.988409902147041637325262574352366540e30
#define LAL_PI 3.14159265359
#define C 299792458.0
#define G 6.67430e-11
#define LAL_MTSUN_SI LAL_MSUN_SI*G/(C*C*C)
#define XLAL_einval  0
#define XLAL_SUCCESS 0



typedef struct tagSpinWeightedSphericalHarmonic_l2 {
   COMPLEX16 Y2m2, Y2m1, Y20, Y21, Y22;
 } SpinWeightedSphericalHarmonic_l2;

const double sqrt_6 = 2.44948974278317788;

double pow_2_of(double x){
  return x*x;
}

 static REAL8 atan2tol(REAL8 a, REAL8 b, REAL8 tol)
 {
   REAL8 c;
   if (fabs(a) < tol && fabs(b) < tol)
     c = 0.;
   else
     c = atan2(a, b);
   return c;
 }

 static REAL8 L2PNR_v1(
   const REAL8 v,   /**< Cubic root of (Pi * Frequency (geometric)) */
   const REAL8 eta) /**< Symmetric mass-ratio */
 {
   const REAL8 mu = eta; /* M=1 */
   const REAL8 v2 = v*v;
   const REAL8 v3 = v2*v;
   const REAL8 v4 = v3*v;
   const REAL8 eta2 = eta*eta;
   const REAL8 b = (4.75 + eta/9.)*eta*v4;
  
  
   return mu*sqrt((1 - ((3 - eta)*v2)/3. + b)/v2)*
     (1 + ((1 - 3*eta)*v2)/2. + (3*(1 - 7*eta + 13*eta2)*v4)/8. +
       ((14 - 41*eta + 4*eta2)*v4)/(4.*pow_2_of(1 - ((3 - eta)*v2)/3. + b)) +
       ((3 + eta)*v2)/(1 - ((3 - eta)*v2)/3. + b) +
       ((7 - 10*eta - 9*eta2)*v4)/(2.*(1 - ((3 - eta)*v2)/3. + b)));
 }

 /**
  * Simple 2PN version of the orbital angular momentum L,
  * without any spin terms expressed as a function of v.
  * For IMRPhenomP(v2).
  *
  *  Reference:
  *  - Boh&eacute; et al, 1212.5520v2 Eq 4.7 first line
  */
 static REAL8 L2PNR(
   const REAL8 v,   /**< Cubic root of (Pi * Frequency (geometric)) */
   const REAL8 eta) /**< Symmetric mass-ratio */
 {
   const REAL8 eta2 = eta*eta;
   const REAL8 x = v*v;
   const REAL8 x2 = x*x;
   return (eta*(1.0 + (1.5 + eta/6.0)*x + (3.375 - (19.0*eta)/8. - eta2/24.0)*x2)) / sqrt(x);
 }

void printstr(char *string){
  for (size_t i = 0; i < strlen(string); i++) {
    // Access each char in the string
    printf("%c",string[i]);
  }
}

void XLAL_ERROR_VAL(int einval, int number){
  printf("\n");
}


void XLAL_ERROR(int einval, char *message){
    printstr(message);
    printf("\n");
}

void XLAL_CHECK(int statement, int fault, char *message){
    if (statement == 0){
      printf("error \n");
      printstr(message);
    }
}


COMPLEX16 XLALSpinWeightedSphericalHarmonic(
                                    REAL8 theta,  /**< polar angle (rad) */
                                    REAL8 phi,    /**< azimuthal angle (rad) */
                                    int s,        /**< spin weight */
                                    int l,        /**< mode number l */
                                    int m         /**< mode number m */
     )
 {
   REAL8 fac;
   COMPLEX16 ans;
  
   /* sanity checks ... */
   if ( l < abs(s) ) 
   {
     XLAL_ERROR_VAL(0, XLAL_einval);
   }
   if ( l < abs(m) ) 
   {
     XLAL_ERROR_VAL(0, XLAL_einval);
   }
  
   if ( s == -2 ) 
   {
     if ( l == 2 ) 
     {
       switch ( m ) 
       {
         case -2:
           fac = sqrt( 5.0 / ( 64.0 * LAL_PI ) ) * ( 1.0 - cos( theta ))*( 1.0 - cos( theta ));
           break;
         case -1:
           fac = sqrt( 5.0 / ( 16.0 * LAL_PI ) ) * sin( theta )*( 1.0 - cos( theta ));
           break;
  
         case 0:
           fac = sqrt( 15.0 / ( 32.0 * LAL_PI ) ) * sin( theta )*sin( theta );
           break;
  
         case 1:
           fac = sqrt( 5.0 / ( 16.0 * LAL_PI ) ) * sin( theta )*( 1.0 + cos( theta ));
           break;
  
         case 2:
           fac = sqrt( 5.0 / ( 64.0 * LAL_PI ) ) * ( 1.0 + cos( theta ))*( 1.0 + cos( theta ));
           break;
         default:
           XLAL_ERROR_VAL(0, XLAL_einval);
           break;
       } /*  switch (m) */
     }  /* l==2*/
     else if ( l == 3 ) 
     {
       switch ( m ) 
       {
         case -3:
           fac = sqrt(21.0/(2.0*LAL_PI))*cos(theta/2.0)*pow(sin(theta/2.0),5.0);
           break;
         case -2:
           fac = sqrt(7.0/(4.0*LAL_PI))*(2.0 + 3.0*cos(theta))*pow(sin(theta/2.0),4.0);
           break;
         case -1:
           fac = sqrt(35.0/(2.0*LAL_PI))*(sin(theta) + 4.0*sin(2.0*theta) - 3.0*sin(3.0*theta))/32.0;
           break;
         case 0:
           fac = (sqrt(105.0/(2.0*LAL_PI))*cos(theta)*pow(sin(theta),2.0))/4.0;
           break;
         case 1:
           fac = -sqrt(35.0/(2.0*LAL_PI))*(sin(theta) - 4.0*sin(2.0*theta) - 3.0*sin(3.0*theta))/32.0;
           break;
  
         case 2:
           fac = sqrt(7.0/LAL_PI)*pow(cos(theta/2.0),4.0)*(-2.0 + 3.0*cos(theta))/2.0;
           break;
  
         case 3:
           fac = -sqrt(21.0/(2.0*LAL_PI))*pow(cos(theta/2.0),5.0)*sin(theta/2.0);
           break;
  
         default:
           XLAL_ERROR_VAL(0, XLAL_einval);
           break;
       }
     }   /* l==3 */
     else if ( l == 4 ) 
     {
       switch ( m ) 
       {
         case -4:
           fac = 3.0*sqrt(7.0/LAL_PI)*pow(cos(theta/2.0),2.0)*pow(sin(theta/2.0),6.0);
           break;
         case -3:
           fac = 3.0*sqrt(7.0/(2.0*LAL_PI))*cos(theta/2.0)*(1.0 + 2.0*cos(theta))*pow(sin(theta/2.0),5.0);
           break;
  
         case -2:
           fac = (3.0*(9.0 + 14.0*cos(theta) + 7.0*cos(2.0*theta))*pow(sin(theta/2.0),4.0))/(4.0*sqrt(LAL_PI));
           break;
         case -1:
           fac = (3.0*(3.0*sin(theta) + 2.0*sin(2.0*theta) + 7.0*sin(3.0*theta) - 7.0*sin(4.0*theta)))/(32.0*sqrt(2.0*LAL_PI));
           break;
         case 0:
           fac = (3.0*sqrt(5.0/(2.0*LAL_PI))*(5.0 + 7.0*cos(2.0*theta))*pow(sin(theta),2.0))/16.0;
           break;
         case 1:
           fac = (3.0*(3.0*sin(theta) - 2.0*sin(2.0*theta) + 7.0*sin(3.0*theta) + 7.0*sin(4.0*theta)))/(32.0*sqrt(2.0*LAL_PI));
           break;
         case 2:
           fac = (3.0*pow(cos(theta/2.0),4.0)*(9.0 - 14.0*cos(theta) + 7.0*cos(2.0*theta)))/(4.0*sqrt(LAL_PI));
           break;
         case 3:
           fac = -3.0*sqrt(7.0/(2.0*LAL_PI))*pow(cos(theta/2.0),5.0)*(-1.0 + 2.0*cos(theta))*sin(theta/2.0);
           break;
         case 4:
           fac = 3.0*sqrt(7.0/LAL_PI)*pow(cos(theta/2.0),6.0)*pow(sin(theta/2.0),2.0);
           break;
         default:
           XLAL_ERROR_VAL(0, XLAL_einval);
           break;
       }
     }    /* l==4 */
     else if ( l == 5 ) 
     {
       switch ( m ) 
       {
         case -5:
           fac = sqrt(330.0/LAL_PI)*pow(cos(theta/2.0),3.0)*pow(sin(theta/2.0),7.0);
           break;
         case -4:
           fac = sqrt(33.0/LAL_PI)*pow(cos(theta/2.0),2.0)*(2.0 + 5.0*cos(theta))*pow(sin(theta/2.0),6.0);
           break;
         case -3:
           fac = (sqrt(33.0/(2.0*LAL_PI))*cos(theta/2.0)*(17.0 + 24.0*cos(theta) + 15.0*cos(2.0*theta))*pow(sin(theta/2.0),5.0))/4.0;
           break;
         case -2:
           fac = (sqrt(11.0/LAL_PI)*(32.0 + 57.0*cos(theta) + 36.0*cos(2.0*theta) + 15.0*cos(3.0*theta))*pow(sin(theta/2.0),4.0))/8.0;
           break;
         case -1:
           fac = (sqrt(77.0/LAL_PI)*(2.0*sin(theta) + 8.0*sin(2.0*theta) + 3.0*sin(3.0*theta) + 12.0*sin(4.0*theta) - 15.0*sin(5.0*theta)))/256.0;
           break;
         case 0:
           fac = (sqrt(1155.0/(2.0*LAL_PI))*(5.0*cos(theta) + 3.0*cos(3.0*theta))*pow(sin(theta),2.0))/32.0;
           break;
         case 1:
           fac = sqrt(77.0/LAL_PI)*(-2.0*sin(theta) + 8.0*sin(2.0*theta) - 3.0*sin(3.0*theta) + 12.0*sin(4.0*theta) + 15.0*sin(5.0*theta))/256.0;
           break;
         case 2:
           fac = sqrt(11.0/LAL_PI)*pow(cos(theta/2.0),4.0)*(-32.0 + 57.0*cos(theta) - 36.0*cos(2.0*theta) + 15.0*cos(3.0*theta))/8.0;
           break;
         case 3:
           fac = -sqrt(33.0/(2.0*LAL_PI))*pow(cos(theta/2.0),5.0)*(17.0 - 24.0*cos(theta) + 15.0*cos(2.0*theta))*sin(theta/2.0)/4.0;
           break;
         case 4:
           fac = sqrt(33.0/LAL_PI)*pow(cos(theta/2.0),6.0)*(-2.0 + 5.0*cos(theta))*pow(sin(theta/2.0),2.0);
           break;
         case 5:
           fac = -sqrt(330.0/LAL_PI)*pow(cos(theta/2.0),7.0)*pow(sin(theta/2.0),3.0);
           break;
         default:
           XLAL_ERROR_VAL(0, XLAL_einval);
           break;
       }
     }  /* l==5 */
     else if ( l == 6 )
     {
       switch ( m )
       {
         case -6:
           fac = (3.*sqrt(715./LAL_PI)*pow(cos(theta/2.0),4)*pow(sin(theta/2.0),8))/2.0;
           break;
         case -5:
           fac = (sqrt(2145./LAL_PI)*pow(cos(theta/2.0),3)*(1. + 3.*cos(theta))*pow(sin(theta/2.0),7))/2.0;
           break;
         case -4:
           fac = (sqrt(195./(2.0*LAL_PI))*pow(cos(theta/2.0),2)*(35. + 44.*cos(theta) 
           + 33.*cos(2.*theta))*pow(sin(theta/2.0),6))/8.0;
           break;
         case -3:
           fac = (3.*sqrt(13./LAL_PI)*cos(theta/2.0)*(98. + 185.*cos(theta) + 110.*cos(2*theta) 
           + 55.*cos(3.*theta))*pow(sin(theta/2.0),5))/32.0;
           break;
         case -2:
           fac = (sqrt(13./LAL_PI)*(1709. + 3096.*cos(theta) + 2340.*cos(2.*theta) + 1320.*cos(3.*theta) 
           + 495.*cos(4.*theta))*pow(sin(theta/2.0),4))/256.0;
           break;
         case -1:
           fac = (sqrt(65./(2.0*LAL_PI))*cos(theta/2.0)*(161. + 252.*cos(theta) + 252.*cos(2.*theta) 
           + 132.*cos(3.*theta) + 99.*cos(4.*theta))*pow(sin(theta/2.0),3))/64.0;
           break;
         case 0:
           fac = (sqrt(1365./LAL_PI)*(35. + 60.*cos(2.*theta) + 33.*cos(4.*theta))*pow(sin(theta),2))/512.0;
           break;
         case 1:
           fac = (sqrt(65./(2.0*LAL_PI))*pow(cos(theta/2.0),3)*(161. - 252.*cos(theta) + 252.*cos(2.*theta) 
           - 132.*cos(3.*theta) + 99.*cos(4.*theta))*sin(theta/2.0))/64.0;
           break;
         case 2:
           fac = (sqrt(13./LAL_PI)*pow(cos(theta/2.0),4)*(1709. - 3096.*cos(theta) + 2340.*cos(2.*theta) 
           - 1320*cos(3*theta) + 495*cos(4*theta)))/256.0;
           break;
         case 3:
           fac = (-3.*sqrt(13./LAL_PI)*pow(cos(theta/2.0),5)*(-98. + 185.*cos(theta) - 110.*cos(2*theta) 
           + 55.*cos(3.*theta))*sin(theta/2.0))/32.0;
           break;
         case 4:
           fac = (sqrt(195./(2.0*LAL_PI))*pow(cos(theta/2.0),6)*(35. - 44.*cos(theta) 
           + 33.*cos(2*theta))*pow(sin(theta/2.0),2))/8.0;
           break;
         case 5:
           fac = -(sqrt(2145./LAL_PI)*pow(cos(theta/2.0),7)*(-1. + 3.*cos(theta))*pow(sin(theta/2.0),3))/2.0;
           break;
         case 6:
           fac = (3.*sqrt(715./LAL_PI)*pow(cos(theta/2.0),8)*pow(sin(theta/2.0),4))/2.0;
           break;
         default:
           XLAL_ERROR_VAL(0, XLAL_einval);
           break;
       }
     } /* l==6 */
     else if ( l == 7 )
     {
       switch ( m )
       {
         case -7:
           fac = sqrt(15015./(2.0*LAL_PI))*pow(cos(theta/2.0),5)*pow(sin(theta/2.0),9);
           break;
         case -6:
           fac = (sqrt(2145./LAL_PI)*pow(cos(theta/2.0),4)*(2. + 7.*cos(theta))*pow(sin(theta/2.0),8))/2.0;
           break;
         case -5:
           fac = (sqrt(165./(2.0*LAL_PI))*pow(cos(theta/2.0),3)*(93. + 104.*cos(theta) 
           + 91.*cos(2.*theta))*pow(sin(theta/2.0),7))/8.0;
           break;
         case -4:
           fac = (sqrt(165./(2.0*LAL_PI))*pow(cos(theta/2.0),2)*(140. + 285.*cos(theta) 
           + 156.*cos(2.*theta) + 91.*cos(3.*theta))*pow(sin(theta/2.0),6))/16.0;
           break;
         case -3:
           fac = (sqrt(15./(2.0*LAL_PI))*cos(theta/2.0)*(3115. + 5456.*cos(theta) + 4268.*cos(2.*theta) 
           + 2288.*cos(3.*theta) + 1001.*cos(4.*theta))*pow(sin(theta/2.0),5))/128.0;
           break;
         case -2:
           fac = (sqrt(15./LAL_PI)*(5220. + 9810.*cos(theta) + 7920.*cos(2.*theta) + 5445.*cos(3.*theta) 
           + 2860.*cos(4.*theta) + 1001.*cos(5.*theta))*pow(sin(theta/2.0),4))/512.0;
           break;
         case -1:
           fac = (3.*sqrt(5./(2.0*LAL_PI))*cos(theta/2.0)*(1890. + 4130.*cos(theta) + 3080.*cos(2.*theta) 
           + 2805.*cos(3.*theta) + 1430.*cos(4.*theta) + 1001.*cos(5*theta))*pow(sin(theta/2.0),3))/512.0;
           break;
         case 0:
           fac = (3.*sqrt(35./LAL_PI)*cos(theta)*(109. + 132.*cos(2.*theta) 
           + 143.*cos(4.*theta))*pow(sin(theta),2))/512.0;
           break;
         case 1:
           fac = (3.*sqrt(5./(2.0*LAL_PI))*pow(cos(theta/2.0),3)*(-1890. + 4130.*cos(theta) - 3080.*cos(2.*theta) 
           + 2805.*cos(3.*theta) - 1430.*cos(4.*theta) + 1001.*cos(5.*theta))*sin(theta/2.0))/512.0;
           break;
         case 2:
           fac = (sqrt(15./LAL_PI)*pow(cos(theta/2.0),4)*(-5220. + 9810.*cos(theta) - 7920.*cos(2.*theta) 
           + 5445.*cos(3.*theta) - 2860.*cos(4.*theta) + 1001.*cos(5.*theta)))/512.0;
           break;
         case 3:
           fac = -(sqrt(15./(2.0*LAL_PI))*pow(cos(theta/2.0),5)*(3115. - 5456.*cos(theta) + 4268.*cos(2.*theta) 
           - 2288.*cos(3.*theta) + 1001.*cos(4.*theta))*sin(theta/2.0))/128.0;
           break;  
         case 4:
           fac = (sqrt(165./(2.0*LAL_PI))*pow(cos(theta/2.0),6)*(-140. + 285.*cos(theta) - 156.*cos(2*theta) 
           + 91.*cos(3.*theta))*pow(sin(theta/2.0),2))/16.0;
           break;
         case 5:
           fac = -(sqrt(165./(2.0*LAL_PI))*pow(cos(theta/2.0),7)*(93. - 104.*cos(theta) 
           + 91.*cos(2.*theta))*pow(sin(theta/2.0),3))/8.0;
           break;
         case 6:
           fac = (sqrt(2145./LAL_PI)*pow(cos(theta/2.0),8)*(-2. + 7.*cos(theta))*pow(sin(theta/2.0),4))/2.0;
           break;
         case 7:
           fac = -(sqrt(15015./(2.0*LAL_PI))*pow(cos(theta/2.0),9)*pow(sin(theta/2.0),5));
           break;
         default:
           XLAL_ERROR_VAL(0, XLAL_einval);
           break;
       }
     } /* l==7 */
     else if ( l == 8 )
     {
       switch ( m )
       {
         case -8:
           fac = sqrt(34034./LAL_PI)*pow(cos(theta/2.0),6)*pow(sin(theta/2.0),10);
           break;
         case -7:
           fac = sqrt(17017./(2.0*LAL_PI))*pow(cos(theta/2.0),5)*(1. + 4.*cos(theta))*pow(sin(theta/2.0),9);
           break;
         case -6:
           fac = sqrt(255255./LAL_PI)*pow(cos(theta/2.0),4)*(1. + 2.*cos(theta))
           *sin(LAL_PI/4.0 - theta/2.0)*sin(LAL_PI/4.0 + theta/2.0)*pow(sin(theta/2.0),8);
           break;
         case -5:
           fac = (sqrt(12155./(2.0*LAL_PI))*pow(cos(theta/2.0),3)*(19. + 42.*cos(theta) 
           + 21.*cos(2.*theta) + 14.*cos(3.*theta))*pow(sin(theta/2.0),7))/8.0;
           break;
         case -4:
           fac = (sqrt(935./(2.0*LAL_PI))*pow(cos(theta/2.0),2)*(265. + 442.*cos(theta) + 364.*cos(2.*theta) 
           + 182.*cos(3.*theta) + 91.*cos(4.*theta))*pow(sin(theta/2.0),6))/32.0;
           break;
         case -3:
           fac = (sqrt(561./(2.0*LAL_PI))*cos(theta/2.0)*(869. + 1660.*cos(theta) + 1300.*cos(2.*theta) 
           + 910.*cos(3.*theta) + 455.*cos(4.*theta) + 182.*cos(5.*theta))*pow(sin(theta/2.0),5))/128.0;
           break;
         case -2:
           fac = (sqrt(17./LAL_PI)*(7626. + 14454.*cos(theta) + 12375.*cos(2.*theta) + 9295.*cos(3.*theta) 
           + 6006.*cos(4.*theta) + 3003.*cos(5.*theta) + 1001.*cos(6.*theta))*pow(sin(theta/2.0),4))/512.0;
           break;
         case -1:
           fac = (sqrt(595./(2.0*LAL_PI))*cos(theta/2.0)*(798. + 1386.*cos(theta) + 1386.*cos(2.*theta) 
           + 1001.*cos(3.*theta) + 858.*cos(4.*theta) + 429.*cos(5.*theta) + 286.*cos(6.*theta))*pow(sin(theta/2.0),3))/512.0;
           break;
         case 0:
           fac = (3.*sqrt(595./LAL_PI)*(210. + 385.*cos(2.*theta) + 286.*cos(4.*theta) 
           + 143.*cos(6.*theta))*pow(sin(theta),2))/4096.0;
           break;
         case 1:
           fac = (sqrt(595./(2.0*LAL_PI))*pow(cos(theta/2.0),3)*(798. - 1386.*cos(theta) + 1386.*cos(2.*theta) 
           - 1001.*cos(3.*theta) + 858.*cos(4.*theta) - 429.*cos(5.*theta) + 286.*cos(6.*theta))*sin(theta/2.0))/512.0;
           break;
         case 2:
           fac = (sqrt(17./LAL_PI)*pow(cos(theta/2.0),4)*(7626. - 14454.*cos(theta) + 12375.*cos(2.*theta) 
           - 9295.*cos(3.*theta) + 6006.*cos(4.*theta) - 3003.*cos(5.*theta) + 1001.*cos(6.*theta)))/512.0;
           break;
         case 3:
           fac = -(sqrt(561./(2.0*LAL_PI))*pow(cos(theta/2.0),5)*(-869. + 1660.*cos(theta) - 1300.*cos(2.*theta) 
           + 910.*cos(3.*theta) - 455.*cos(4.*theta) + 182.*cos(5.*theta))*sin(theta/2.0))/128.0;
           break;
         case 4:
           fac = (sqrt(935./(2.0*LAL_PI))*pow(cos(theta/2.0),6)*(265. - 442.*cos(theta) + 364.*cos(2.*theta) 
           - 182.*cos(3.*theta) + 91.*cos(4.*theta))*pow(sin(theta/2.0),2))/32.0;
           break;
         case 5:
           fac = -(sqrt(12155./(2.0*LAL_PI))*pow(cos(theta/2.0),7)*(-19. + 42.*cos(theta) - 21.*cos(2.*theta) 
           + 14.*cos(3.*theta))*pow(sin(theta/2.0),3))/8.0;
           break;
         case 6:
           fac = sqrt(255255./LAL_PI)*pow(cos(theta/2.0),8)*(-1. + 2.*cos(theta))*sin(LAL_PI/4.0 - theta/2.0)
           *sin(LAL_PI/4.0 + theta/2.0)*pow(sin(theta/2.0),4);
           break;
         case 7:
           fac = -(sqrt(17017./(2.0*LAL_PI))*pow(cos(theta/2.0),9)*(-1. + 4.*cos(theta))*pow(sin(theta/2.0),5));
           break;
         case 8:
           fac = sqrt(34034./LAL_PI)*pow(cos(theta/2.0),10)*pow(sin(theta/2.0),6);
           break;
         default:
           XLAL_ERROR_VAL(0, XLAL_einval);
           break;
       }
     } /* l==8 */
     else 
     {
       XLAL_ERROR_VAL(0, XLAL_einval);
     }
   }
   else 
   {
     XLAL_ERROR_VAL(0, XLAL_einval);
   }
   if (m)
     ans = cexp(1I*m*phi) * fac;
   else
     ans = fac;
   return ans;
 }

typedef enum tagIMRPhenomP_version_type {
  IMRPhenomPv1_V, /**< version 1: based on IMRPhenomC */
  IMRPhenomPv2_V,  /**< version 2: based on IMRPhenomD */
  IMRPhenomPv2NRTidal_V, /**< version Pv2_NRTidal: based on IMRPhenomPv2; NRTides added before precession; can be used with both NRTidal versions defined below */
  IMRPhenomPv3_V  /**< version 3: based on IMRPhenomD and the precession angles from Katerina Chatziioannou PhysRevD.95.104004 (arxiv:1703.03967) */
 } IMRPhenomP_version_type;

/**
  * Function to map LAL parameters
  * (masses, 6 spin components, phiRef and inclination at f_ref)
  * (assumed to be in the source frame
  *  where LN points in the z direction
  *  i.e. lnhat = (0,0,1)
  *  and the separation vector n is in the x direction
  *  and the spherical angles of the line of sight N are (incl,Pi/2-phiRef))
  * into IMRPhenomP intrinsic parameters
  * (chi1_l, chi2_l, chip, thetaJN, alpha0 and phi_aligned).
  *
  * All input masses and frequencies should be in SI units.
  *
  * See Fig. 1. in arxiv:1408.1810 for a diagram of the angles.
  */
 int XLALSimIMRPhenomPCalculateModelParametersFromSourceFrame(
     REAL8 *chi1_l,                  /**< [out] Dimensionless aligned spin on companion 1 */
     REAL8 *chi2_l,                  /**< [out] Dimensionless aligned spin on companion 2 */
     REAL8 *chip,                    /**< [out] Effective spin in the orbital plane */
     REAL8 *thetaJN,                  /**< [out] Angle between J0 and line of sight (z-direction) */
     REAL8 *alpha0,                  /**< [out] Initial value of alpha angle (azimuthal precession angle) */
     REAL8 *phi_aligned,                  /**< [out] Initial phase to feed the underlying aligned-spin model */
     REAL8 *zeta_polariz,                  /**< [out] Angle to rotate the polarizations */
     const REAL8 m1_SI,              /**< Mass of companion 1 (kg) */
     const REAL8 m2_SI,              /**< Mass of companion 2 (kg) */
     const REAL8 f_ref,              /**< Reference GW frequency (Hz) */
     const REAL8 phiRef,              /**< Reference phase */
     const REAL8 incl,              /**< Inclination : angle between LN and the line of sight */
     const REAL8 s1x,                /**< Initial value of s1x: dimensionless spin of BH 1 */
     const REAL8 s1y,                /**< Initial value of s1y: dimensionless spin of BH 1 */
     const REAL8 s1z,                /**< Initial value of s1z: dimensionless spin of BH 1 */
     const REAL8 s2x,                /**< Initial value of s2x: dimensionless spin of BH 2 */
     const REAL8 s2y,                /**< Initial value of s2y: dimensionless spin of BH 2 */
     const REAL8 s2z,                /**< Initial value of s2z: dimensionless spin of  BH 2 */
     IMRPhenomP_version_type IMRPhenomP_version /**< IMRPhenomP(v1) uses IMRPhenomC, IMRPhenomPv2 uses IMRPhenomD, IMRPhenomPv2_NRTidal uses NRTidal framework with IMRPhenomPv2 */
 )
 {
   // Note that the angle phiJ defined below and alpha0 are degenerate. Therefore we do not output phiJ.
  
   /* Check arguments for sanity */
   XLAL_CHECK(chi1_l != NULL, XLAL_EFAULT,"");
   XLAL_CHECK(chi2_l != NULL, XLAL_EFAULT,"");
   XLAL_CHECK(chip != NULL, XLAL_EFAULT,"");
   XLAL_CHECK(thetaJN != NULL, XLAL_EFAULT,"");
   XLAL_CHECK(alpha0 != NULL, XLAL_EFAULT,"");
   XLAL_CHECK(phi_aligned != NULL, XLAL_EFAULT,"");
  
   XLAL_CHECK(f_ref > 0, XLAL_EDOM, "Reference frequency must be positive.\n");
   XLAL_CHECK(m1_SI > 0, XLAL_EDOM, "m1 must be positive.\n");
   XLAL_CHECK(m2_SI > 0, XLAL_EDOM, "m2 must be positive.\n");
   XLAL_CHECK(fabs(s1x*s1x + s1y*s1y + s1z*s1z) <= 1.0, XLAL_EDOM, "|S1/m1^2| must be <= 1.\n");
   XLAL_CHECK(fabs(s2x*s2x + s2y*s2y + s2z*s2z) <= 1.0, XLAL_EDOM, "|S2/m2^2| must be <= 1.\n");
  
   const REAL8 m1 = m1_SI / LAL_MSUN_SI;   /* Masses in solar masses */
   const REAL8 m2 = m2_SI / LAL_MSUN_SI;
   const REAL8 M = m1+m2;
   const REAL8 m1_2 = m1*m1;
   const REAL8 m2_2 = m2*m2;
   const REAL8 eta = m1 * m2 / (M*M);    /* Symmetric mass-ratio */
  
   /* From the components in the source frame, we can easily determine
    chi1_l, chi2_l, chip and phi_aligned, which we need to return.
    We also compute the spherical angles of J,
    which we need to transform to the J frame*/
  
   /* Aligned spins */
   *chi1_l = s1z; /* Dimensionless aligned spin on BH 1 */
   *chi2_l = s2z; /* Dimensionless aligned spin on BH 2 */
  
   /* Magnitude of the spin projections in the orbital plane */
   const REAL8 S1_perp = m1_2*sqrt(s1x*s1x + s1y*s1y);
   const REAL8 S2_perp = m2_2*sqrt(s2x*s2x + s2y*s2y);

   //printf("perps: %lf %lf \n", S1_perp, S2_perp);

   /* From this we can compute chip*/
   const REAL8 A1 = 2 + (3*m2) / (2*m1);
   const REAL8 A2 = 2 + (3*m1) / (2*m2);
   const REAL8 ASp1 = A1*S1_perp;
   const REAL8 ASp2 = A2*S2_perp;
   const REAL8 num = (ASp2 > ASp1) ? ASp2 : ASp1;
   const REAL8 den = (m2 > m1) ? A2*m2_2 : A1*m1_2;
   *chip = num / den; /*  chip = max(A1 Sp1, A2 Sp2) / (A_i m_i^2) for i index of larger BH (See Eqn. 32 in technical document) */
  
   /* Compute L, J0 and orientation angles */
   const REAL8 m_sec = M * LAL_MTSUN_SI;   /* Total mass in seconds */
   const REAL8 piM = LAL_PI * m_sec;
   //printf("piM: %lf \n", piM );
   const REAL8 v_ref = cbrt(piM * f_ref);
  
   const int ExpansionOrder = 5; // Used in PhenomPv3 only
  
   REAL8 L0 = 0.0;
   switch (IMRPhenomP_version) {
     case IMRPhenomPv1_V:
       printf("I am here v1\n");

       L0 = M*M * L2PNR_v1(v_ref, eta); /* Use 2PN approximation for L. */
       break;
     case IMRPhenomPv2_V:
     case IMRPhenomPv2NRTidal_V:
       printf("I am here v2\n");
       L0 = M*M * L2PNR(v_ref, eta);   /* Use 2PN approximation for L. */
       break;
     case IMRPhenomPv3_V: /*Pv3 uses 3PN spinning for L but in non-precessing limit uses the simpler L2PNR function */
       printf("I am here v3\n");
       if ((s1x == 0. && s1y == 0. && s2x == 0. && s2y == 0.))
       { // non-precessing case
         L0 = M * M * L2PNR(v_ref, eta); /* Use 2PN approximation for L. */
       } else { // precessing case
         //L0 = M * M * PhenomInternal_OrbAngMom3PN(f_ref / 2., m1_SI, m2_SI, s1x, s1y, s1z, s2x, s2y, s2z, f_ref, ExpansionOrder); /* Use 3PN spinning approximation for L. */
          L0=0;
       }
       break;
     default:
       XLAL_ERROR( XLAL_einval, "Unknown IMRPhenomP version!\nAt present only v1 and v2 are available." );
       break;
     }
   //printf("L0 input: %.10f %.10f %.10f \n", v_ref, eta, M);
   //printf("L0: %.10f \n", L0);
   // Below, _sf indicates source frame components. We will also use _Jf for J frame components
   const REAL8 J0x_sf = m1_2*s1x + m2_2*s2x;
   const REAL8 J0y_sf = m1_2*s1y + m2_2*s2y;
   const REAL8 J0z_sf = L0 + m1_2*s1z + m2_2*s2z;
   const REAL8 J0 = sqrt(J0x_sf*J0x_sf + J0y_sf*J0y_sf + J0z_sf*J0z_sf);
  
   /* Compute thetaJ, the angle between J0 and LN (z-direction) */
   REAL8 thetaJ_sf;
   if (J0 < 1e-10) {
     //XLAL_PRINT_WARNING("Warning: |J0| < 1e-10. Setting thetaJ = 0.\n");
     //printf("Warning: |J0| < 1e-10. Setting thetaJ = 0.\n");
     thetaJ_sf = 0;
   } else {
     thetaJ_sf = acos(J0z_sf / J0);
   }
  
   REAL8 phiJ_sf;
   if (fabs(J0x_sf) < MAX_TOL_ATAN && fabs(J0y_sf) < MAX_TOL_ATAN)
     phiJ_sf = LAL_PI/2. - phiRef; // aligned spin limit
   else
     phiJ_sf = atan2(J0y_sf, J0x_sf); /* azimuthal angle of J0 in the source frame */
  
   *phi_aligned = - phiJ_sf;
  
   /* We now have to rotate to the "J frame" where we can easily
    compute alpha0, the azimuthal angle of LN,
    as well as thetaJ, the angle between J and N.
    The J frame is defined imposing that J points in the z direction
    and the line of sight N is in the xz plane (with positive projection along x).
    The components of any vector in the (new) J frame are obtained from those
    in the (old) source frame by multiplying by RZ[kappa].RY[-thetaJ].RZ[-phiJ]
    where kappa will be determined by rotating N with RY[-thetaJ].RZ[-phiJ]
    (which brings J to the z axis) and taking the opposite of azimuthal angle of the rotated N.
    */
   REAL8 tmp1,tmp2;
   // First we determine kappa
   // in the source frame, the components of N are given in Eq (35c) of T1500606-v6
   REAL8 Nx_sf = sin(incl)*cos(LAL_PI/2. - phiRef);
   REAL8 Ny_sf = sin(incl)*sin(LAL_PI/2. - phiRef);
   REAL8 Nz_sf = cos(incl);
   REAL8 tmp_x = Nx_sf;
   REAL8 tmp_y = Ny_sf;
   REAL8 tmp_z = Nz_sf;
   ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z);
   ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z);
   REAL8 kappa;
   kappa = - atan2tol(tmp_y,tmp_x, MAX_TOL_ATAN);
   //printf("%lf,\n",kappa);
  
   // Then we determine alpha0, by rotating LN
   tmp_x = 0.;
   tmp_y = 0.;
   tmp_z = 1.; // in the source frame, LN=(0,0,1)
   ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z);
   ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z);
   ROTATEZ(kappa, tmp_x, tmp_y, tmp_z);
   //printf("%lf, %lf \n",tmp_x,tmp_y);
   if (fabs(tmp_x) < MAX_TOL_ATAN && fabs(tmp_y) < MAX_TOL_ATAN)
     *alpha0 = LAL_PI; //this is the aligned spin case
   else
     *alpha0 = atan2(tmp_y,tmp_x);
  
   // Finally we determine thetaJ, by rotating N
   tmp_x = Nx_sf;
   tmp_y = Ny_sf;
   tmp_z = Nz_sf;
   ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z);
   ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z);
   ROTATEZ(kappa, tmp_x, tmp_y, tmp_z);
   REAL8 Nx_Jf = tmp_x; // let's store those two since we will reuse them later (we don't need the y component)
   REAL8 Nz_Jf = tmp_z;
   *thetaJN = acos(Nz_Jf); // No normalization needed, we are dealing with a unit vector
  
   /* Finally, we need to redefine the polarizations :
    PhenomP's polarizations are defined following Arun et al (arXiv:0810.5336)
    i.e. projecting the metric onto the P,Q,N triad defined with P=NxJ/|NxJ| (see (2.6) in there).
    By contrast, the triad X,Y,N used in LAL
    ("waveframe" in the nomenclature of T1500606-v6)
    is defined in e.g. eq (35) of this document
    (via its components in the source frame; note we use the defautl Omega=Pi/2).
    Both triads differ from each other by a rotation around N by an angle \zeta
    and we need to rotate the polarizations accordingly by 2\zeta
   */
   REAL8 Xx_sf = -cos(incl)*sin(phiRef);
   REAL8 Xy_sf = -cos(incl)*cos(phiRef);
   REAL8 Xz_sf = sin(incl);
   tmp_x = Xx_sf;
   tmp_y = Xy_sf;
   tmp_z = Xz_sf;
   ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z);
   ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z);
   ROTATEZ(kappa, tmp_x, tmp_y, tmp_z);
   //now the tmp_a are the components of X in the J frame
   //we need the polar angle of that vector in the P,Q basis of Arun et al
   // P=NxJ/|NxJ| and since we put N in the (pos x)z half plane of the J frame
   REAL8 PArunx_Jf = 0.;
   REAL8 PAruny_Jf = -1.;
   REAL8 PArunz_Jf = 0.;
   // Q=NxP
   REAL8 QArunx_Jf = Nz_Jf;
   REAL8 QAruny_Jf = 0.;
   REAL8 QArunz_Jf = -Nx_Jf;
   REAL8 XdotPArun = tmp_x*PArunx_Jf+tmp_y*PAruny_Jf+tmp_z*PArunz_Jf;
   REAL8 XdotQArun = tmp_x*QArunx_Jf+tmp_y*QAruny_Jf+tmp_z*QArunz_Jf;
   *zeta_polariz = atan2(XdotQArun , XdotPArun);
  
   return XLAL_SUCCESS;
 }


 typedef struct tagNNLOanglecoeffs {
     REAL8 alphacoeff1; /* Coefficient of omega^(-1)   in alphaNNLO */
     REAL8 alphacoeff2; /* Coefficient of omega^(-2/3) in alphaNNLO */
     REAL8 alphacoeff3; /* Coefficient of omega^(-1/3) in alphaNNLO */
     REAL8 alphacoeff4; /* Coefficient of log(omega)   in alphaNNLO */
     REAL8 alphacoeff5; /* Coefficient of omega^(1/3)  in alphaNNLO */
  
     REAL8 epsiloncoeff1; /* Coefficient of omega^(-1)   in epsilonNNLO */
     REAL8 epsiloncoeff2; /* Coefficient of omega^(-2/3) in epsilonNNLO */
     REAL8 epsiloncoeff3; /* Coefficient of omega^(-1/3) in epsilonNNLO */
     REAL8 epsiloncoeff4; /* Coefficient of log(omega)   in epsilonNNLO */
     REAL8 epsiloncoeff5; /* Coefficient of omega^(1/3)  in epsilonNNLO */
 } NNLOanglecoeffs;
 /**
  * Next-to-next-to-leading order PN coefficients
  * for Euler angles \f$\alpha\f$ and \f$\epsilon\f$.
  */

 static void ComputeNNLOanglecoeffs(
   NNLOanglecoeffs *angcoeffs, /**< [out] Structure to store results */
   const REAL8 q,              /**< Mass-ratio (convention q>1) */
   const REAL8 chil,           /**< Dimensionless aligned spin of the largest BH */
   const REAL8 chip)           /**< Dimensionless spin component in the orbital plane */
 {
   const REAL8 m2 = q/(1. + q);
   const REAL8 m1 = 1./(1. + q);
   const REAL8 dm = m1 - m2;
   const REAL8 mtot = 1.;
   const REAL8 eta = m1*m2; /* mtot = 1 */
   const REAL8 eta2 = eta*eta;
   const REAL8 eta3 = eta2*eta;
   const REAL8 eta4 = eta3*eta;
   const REAL8 mtot2 = mtot*mtot;
   const REAL8 mtot4 = mtot2*mtot2;
   const REAL8 mtot6 = mtot4*mtot2;
   const REAL8 mtot8 = mtot6*mtot2;
   const REAL8 chil2 = chil*chil;
   const REAL8 chip2 = chip*chip;
   const REAL8 chip4 = chip2*chip2;
   const REAL8 dm2 = dm*dm;
   const REAL8 dm3 = dm2*dm;
   const REAL8 m2_2 = m2*m2;
   const REAL8 m2_3 = m2_2*m2;
   const REAL8 m2_4 = m2_3*m2;
   const REAL8 m2_5 = m2_4*m2;
   const REAL8 m2_6 = m2_5*m2;
   const REAL8 m2_7 = m2_6*m2;
   const REAL8 m2_8 = m2_7*m2;
  
  
   angcoeffs->alphacoeff1 = (-0.18229166666666666 - (5*dm)/(64.*m2));
  
   angcoeffs->alphacoeff2 = ((-15*dm*m2*chil)/(128.*mtot2*eta) - (35*m2_2*chil)/(128.*mtot2*eta));
  
   angcoeffs->alphacoeff3 = (-1.7952473958333333 - (4555*dm)/(7168.*m2) -
         (15*chip2*dm*m2_3)/(128.*mtot4*eta2) -
         (35*chip2*m2_4)/(128.*mtot4*eta2) - (515*eta)/384. - (15*dm2*eta)/(256.*m2_2) -
         (175*dm*eta)/(256.*m2));
  
   angcoeffs->alphacoeff4 = - (35*LAL_PI)/48. - (5*dm*LAL_PI)/(16.*m2) +
      (5*dm2*chil)/(16.*mtot2) + (5*dm*m2*chil)/(3.*mtot2) +
      (2545*m2_2*chil)/(1152.*mtot2) -
      (5*chip2*dm*m2_5*chil)/(128.*mtot6*eta3) -
      (35*chip2*m2_6*chil)/(384.*mtot6*eta3) + (2035*dm*m2*chil)/(21504.*mtot2*eta) +
      (2995*m2_2*chil)/(9216.*mtot2*eta);
  
   angcoeffs->alphacoeff5 = (4.318908476114694 + (27895885*dm)/(2.1676032e7*m2) -
         (15*chip4*dm*m2_7)/(512.*mtot8*eta4) -
         (35*chip4*m2_8)/(512.*mtot8*eta4) -
         (485*chip2*dm*m2_3)/(14336.*mtot4*eta2) +
         (475*chip2*m2_4)/(6144.*mtot4*eta2) +
         (15*chip2*dm2*m2_2)/(256.*mtot4*eta) + (145*chip2*dm*m2_3)/(512.*mtot4*eta) +
         (575*chip2*m2_4)/(1536.*mtot4*eta) + (39695*eta)/86016. + (1615*dm2*eta)/(28672.*m2_2) -
         (265*dm*eta)/(14336.*m2) + (955*eta2)/576. + (15*dm3*eta2)/(1024.*m2_3) +
         (35*dm2*eta2)/(256.*m2_2) + (2725*dm*eta2)/(3072.*m2) - (15*dm*m2*LAL_PI*chil)/(16.*mtot2*eta) -
         (35*m2_2*LAL_PI*chil)/(16.*mtot2*eta) + (15*chip2*dm*m2_7*chil2)/(128.*mtot8*eta4) +
         (35*chip2*m2_8*chil2)/(128.*mtot8*eta4) +
         (375*dm2*m2_2*chil2)/(256.*mtot4*eta) + (1815*dm*m2_3*chil2)/(256.*mtot4*eta) +
         (1645*m2_4*chil2)/(192.*mtot4*eta));
  
   angcoeffs->epsiloncoeff1 = (-0.18229166666666666 - (5*dm)/(64.*m2));
  
   angcoeffs->epsiloncoeff2 = ((-15*dm*m2*chil)/(128.*mtot2*eta) - (35*m2_2*chil)/(128.*mtot2*eta));
  
   angcoeffs->epsiloncoeff3 = (-1.7952473958333333 - (4555*dm)/(7168.*m2) - (515*eta)/384. -
         (15*dm2*eta)/(256.*m2_2) - (175*dm*eta)/(256.*m2));
  
   angcoeffs->epsiloncoeff4 = - (35*LAL_PI)/48. - (5*dm*LAL_PI)/(16.*m2) +
      (5*dm2*chil)/(16.*mtot2) + (5*dm*m2*chil)/(3.*mtot2) +
      (2545*m2_2*chil)/(1152.*mtot2) + (2035*dm*m2*chil)/(21504.*mtot2*eta) +
      (2995*m2_2*chil)/(9216.*mtot2*eta);
  
   angcoeffs->epsiloncoeff5 = (4.318908476114694 + (27895885*dm)/(2.1676032e7*m2) + (39695*eta)/86016. +
         (1615*dm2*eta)/(28672.*m2_2) - (265*dm*eta)/(14336.*m2) + (955*eta2)/576. +
         (15*dm3*eta2)/(1024.*m2_3) + (35*dm2*eta2)/(256.*m2_2) +
         (2725*dm*eta2)/(3072.*m2) - (15*dm*m2*LAL_PI*chil)/(16.*mtot2*eta) - (35*m2_2*LAL_PI*chil)/(16.*mtot2*eta) +
         (375*dm2*m2_2*chil2)/(256.*mtot4*eta) + (1815*dm*m2_3*chil2)/(256.*mtot4*eta) +
         (1645*m2_4*chil2)/(192.*mtot4*eta));
 }

 static void WignerdCoefficients_SmallAngleApproximation(
   REAL8 *cos_beta_half, /**< Output: cos(beta/2) */
   REAL8 *sin_beta_half, /**< Output: sin(beta/2) */
   const REAL8 v,        /**< Cubic root of (Pi * Frequency (geometric)) */
   const REAL8 SL,       /**< Dimensionfull aligned spin */
   const REAL8 eta,      /**< Symmetric mass-ratio */
   const REAL8 Sp)       /**< Dimensionfull spin component in the orbital plane */
 {
   //XLAL_CHECK_VOID(cos_beta_half != NULL, XLAL_EFAULT);
   //XLAL_CHECK_VOID(sin_beta_half != NULL, XLAL_EFAULT);
   REAL8 s = Sp / (L2PNR_v1(v, eta) + SL);  /* s := Sp / (L + SL) */
   REAL8 s2 = s*s;
   *cos_beta_half = 1.0/sqrt(1.0 + s2/4.0);           /* cos(beta/2) */
   *sin_beta_half = sqrt(1.0 - 1.0/(1.0 + s2/4.0));   /* sin(beta/2) */
 }

  static void WignerdCoefficients(
   REAL8 *cos_beta_half, /**< [out] cos(beta/2) */
   REAL8 *sin_beta_half, /**< [out] sin(beta/2) */
   const REAL8 v,        /**< Cubic root of (Pi * Frequency (geometric)) */
   const REAL8 SL,       /**< Dimensionfull aligned spin */
   const REAL8 eta,      /**< Symmetric mass-ratio */
   const REAL8 Sp)       /**< Dimensionfull spin component in the orbital plane */
 {
   //XLAL_CHECK_VOID(cos_beta_half != NULL, XLAL_EFAULT);
   //XLAL_CHECK_VOID(sin_beta_half != NULL, XLAL_EFAULT);
   /* We define the shorthand s := Sp / (L + SL) */
   const REAL8 L = L2PNR(v, eta);
     // We ignore the sign of L + SL below.
   REAL8 s = Sp / (L + SL);  /* s := Sp / (L + SL) */
   REAL8 s2 = s*s;
   REAL8 cos_beta = 1.0 / sqrt(1.0 + s2);
   *cos_beta_half = + sqrt( (1.0 + cos_beta) / 2.0 );  /* cos(beta/2) */
   *sin_beta_half = + sqrt( (1.0 - cos_beta) / 2.0 );  /* sin(beta/2) */
 }

 static int PhenomPCoreTwistUp(
   const REAL8 fHz,                            /**< Frequency (Hz) */
   COMPLEX16 hPhenom,                    /**< [in] IMRPhenom waveform (before precession) */
   const REAL8 eta,                            /**< Symmetric mass ratio */
   const REAL8 chi1_l,                         /**< Dimensionless aligned spin on companion 1 */
   const REAL8 chi2_l,                         /**< Dimensionless aligned spin on companion 2 */
   const REAL8 chip,                           /**< Dimensionless spin in the orbital plane */
   const REAL8 M,                              /**< Total mass (Solar masses) */
   NNLOanglecoeffs *angcoeffs,                 /**< Struct with PN coeffs for the NNLO angles */
   SpinWeightedSphericalHarmonic_l2 *Y2m,      /**< Struct of l=2 spherical harmonics of spin weight -2 */
   const REAL8 alphaoffset,                    /**< f_ref dependent offset for alpha angle (azimuthal precession angle) */
   const REAL8 epsilonoffset,                  /**< f_ref dependent offset for epsilon angle */
   COMPLEX16 *hp,                              /**< [out] plus polarization \f$\tilde h_+\f$ */
   COMPLEX16 *hc,                              /**< [out] cross polarization \f$\tilde h_x\f$ */
   IMRPhenomP_version_type IMRPhenomP_version  /**< IMRPhenomP(v1) uses IMRPhenomC, IMRPhenomPv2 uses IMRPhenomD, IMRPhenomPv2_NRTidal uses NRTidal framework with IMRPhenomPv2 */
 )
 {
  printf("\n\nWhat is even my hPhenom?\n");
   printf("temp: %.10f + %.10fj \n", 
        creal( hPhenom), cimag( hPhenom));
   XLAL_CHECK(angcoeffs != NULL, XLAL_EFAULT,"");
   XLAL_CHECK(hp != NULL, XLAL_EFAULT,"");
   XLAL_CHECK(hc != NULL, XLAL_EFAULT,"");
   XLAL_CHECK(Y2m != NULL, XLAL_EFAULT,"");
  
   REAL8 f = fHz*LAL_MTSUN_SI*M; /* Frequency in geometric units */
  
   const REAL8 q = (1.0 + sqrt(1.0 - 4.0*eta) - 2.0*eta)/(2.0*eta);
   const REAL8 m1 = 1.0/(1.0+q);       /* Mass of the smaller BH for unit total mass M=1. */
   const REAL8 m2 = q/(1.0+q);         /* Mass of the larger BH for unit total mass M=1. */
   const REAL8 Sperp = chip*(m2*m2);   /* Dimensionfull spin component in the orbital plane. S_perp = S_2_perp */
   REAL8 SL;                           /* Dimensionfull aligned spin. */
   const REAL8 chi_eff = (m1*chi1_l + m2*chi2_l); /* effective spin for M=1 */
  
   /* Calculate dimensionfull spins */
   switch (IMRPhenomP_version) {
     case IMRPhenomPv1_V:
       SL = chi_eff*m2;        /* Dimensionfull aligned spin of the largest BH. SL = m2^2 chil = m2*M*chi_eff */
       break;
     case IMRPhenomPv2_V:
     case IMRPhenomPv2NRTidal_V:
       SL = chi1_l*m1*m1 + chi2_l*m2*m2;        /* Dimensionfull aligned spin. */
       break;    default:
       XLAL_ERROR( XLAL_einval, "Unknown IMRPhenomP version!\nAt present only v1 and v2 and tidal are available." );
       break;
   }
  
   /* Compute PN NNLO angles */
   const REAL8 omega = LAL_PI * f;
   const REAL8 logomega = log(omega);
   const REAL8 omega_cbrt = cbrt(omega);
   const REAL8 omega_cbrt2 = omega_cbrt*omega_cbrt;
  
   //printf("step check: %.10f, %.10f, %.10f, %.10f \n", omega, logomega, omega_cbrt, omega_cbrt2);
   REAL8 alpha = (angcoeffs->alphacoeff1/omega
               + angcoeffs->alphacoeff2/omega_cbrt2
               + angcoeffs->alphacoeff3/omega_cbrt
               + angcoeffs->alphacoeff4*logomega
               + angcoeffs->alphacoeff5*omega_cbrt) - alphaoffset;
  
   REAL8 epsilon = (angcoeffs->epsiloncoeff1/omega
                 + angcoeffs->epsiloncoeff2/omega_cbrt2
                 + angcoeffs->epsiloncoeff3/omega_cbrt
                 + angcoeffs->epsiloncoeff4*logomega
                 + angcoeffs->epsiloncoeff5*omega_cbrt) - epsilonoffset;
  
    printf("\n\nalpha epsilon: %.10f, %.10f\n\n", alpha, epsilon);
   /* Calculate intermediate expressions cos(beta/2), sin(beta/2) and powers thereof for Wigner d's. */
   REAL8 cBetah, sBetah; /* cos(beta/2), sin(beta/2) */
   switch (IMRPhenomP_version) {
     case IMRPhenomPv1_V:
       WignerdCoefficients_SmallAngleApproximation(&cBetah, &sBetah, omega_cbrt, SL, eta, Sperp);
       break;
     case IMRPhenomPv2_V:
     case IMRPhenomPv2NRTidal_V:
       WignerdCoefficients(&cBetah, &sBetah, omega_cbrt, SL, eta, Sperp);
       break;
   default:
     XLAL_ERROR( XLAL_einval, " Unknown IMRPhenomP version!\nAt present only v1 and v2 and tidal are available." );
     break;
   }
  
   const REAL8 cBetah2 = cBetah*cBetah;
   const REAL8 cBetah3 = cBetah2*cBetah;
   const REAL8 cBetah4 = cBetah3*cBetah;
   const REAL8 sBetah2 = sBetah*sBetah;
   const REAL8 sBetah3 = sBetah2*sBetah;
   const REAL8 sBetah4 = sBetah3*sBetah;
  
   /* Compute Wigner d coefficients
     The expressions below agree with refX [Goldstein?] and Mathematica
     d2  = Table[WignerD[{2, mp, 2}, 0, -\[Beta], 0], {mp, -2, 2}]
     dm2 = Table[WignerD[{2, mp, -2}, 0, -\[Beta], 0], {mp, -2, 2}]
   */
   COMPLEX16 d2[5]   = {sBetah4, 2*cBetah*sBetah3, sqrt_6*sBetah2*cBetah2, 2*cBetah3*sBetah, cBetah4};
   COMPLEX16 dm2[5]  = {d2[4], -d2[3], d2[2], -d2[1], d2[0]}; /* Exploit symmetry d^2_{-2,-m} = (-1)^m d^2_{2,m} */
  
    //printf("Y2mA: %.10f, %.10f, %.10f, %.10f, %.10f \n", creal(Y2m->Y2m2), creal(Y2m->Y2m1), creal(Y2m->Y20), creal(Y2m->Y21),creal(Y2m->Y22));
   COMPLEX16 Y2mA[5] = {Y2m->Y2m2, Y2m->Y2m1, Y2m->Y20, Y2m->Y21, Y2m->Y22};
   //printf("dm2: %.10f, %.10f, %.10f, %.10f, %.10f \n", creal(dm2[0]), creal(dm2[1]), creal(dm2[2]), creal(dm2[3]),creal(dm2[4]));
   //printf("dm2imag: %.10f, %.10f, %.10f, %.10f, %.10f \n", cimag(dm2[0]), cimag(dm2[1]), cimag(dm2[2]), cimag(dm2[3]),cimag(dm2[4]));
    //printf("Y2mA: %.10f, %.10f, %.10f, %.10f, %.10f \n", creal(Y2mA[0]), creal(Y2mA[1]), creal(Y2mA[2]), creal(Y2mA[3]),creal(Y2mA[4]));
   //printf("\n\ndm2[4]: %.10f + i %.10f \n\n", 
   //         creal(dm2[4]), cimag(dm2[4]));
   
   COMPLEX16 hp_sum = 0;
   COMPLEX16 hc_sum = 0;
  
   /* Sum up contributions to \tilde h+ and \tilde hx */
   /* Precompute powers of e^{i m alpha} */
   COMPLEX16 cexp_i_alpha = cexp(+I*alpha);
   COMPLEX16 cexp_2i_alpha = cexp_i_alpha*cexp_i_alpha;
   COMPLEX16 cexp_mi_alpha = 1.0/cexp_i_alpha;
   COMPLEX16 cexp_m2i_alpha = cexp_mi_alpha*cexp_mi_alpha;
   COMPLEX16 cexp_im_alpha[5] = {cexp_m2i_alpha, cexp_mi_alpha, 1.0, cexp_i_alpha, cexp_2i_alpha};
   //printf("epsilon: %.10f + %.10fj \n", creal(epsilon), cimag(epsilon) );

   
   for(int m=-2; m<=2; m++) {
     COMPLEX16 T2m   = cexp_im_alpha[-m+2] * dm2[m+2] *      Y2mA[m+2];  /*  = 
     cexp(-I*m*alpha) * dm2[m+2] *      Y2mA[m+2] */
     //printf("T2m: %.10f + %.10fj \n", creal(T2m), cimag(T2m));
     COMPLEX16 Tm2m  = cexp_im_alpha[m+2]  * d2[m+2]  * conj(Y2mA[m+2]); /*  = cexp(+I*m*alpha) * d2[m+2]  * conj(Y2mA[m+2]) */
     //printf("Tm2m: %.10f + %.10fj \n", creal(T2m), cimag(T2m));
     hp_sum +=     T2m + Tm2m;
     hc_sum += +I*(T2m - Tm2m);
   }
    printf("hp_sum: %.10f + %.10fj \n", creal(hp_sum), cimag(hp_sum));
  
   COMPLEX16 eps_phase_hP = cexp(-2*I*epsilon) * hPhenom / 2.0;
    printf("temp: %.10f + %.10fj \n", 
        creal( hPhenom), cimag( hPhenom));

    printf("eps_phase: %.10f + %.10fj \n", creal(eps_phase_hP), cimag(eps_phase_hP));
   *hp = eps_phase_hP * hp_sum;
   *hc = eps_phase_hP * hc_sum;
  
   return XLAL_SUCCESS;
  
 }


 // Final Spin and Radiated Energy formulas described in 1508.07250
  
 /**
  * Formula to predict the final spin. Equation 3.6 arXiv:1508.07250
  * s defined around Equation 3.6.
  */
 static double FinalSpin0815_s(double eta, double s) {
   double eta2 = eta*eta;
   double eta3 = eta2*eta;
   double s2 = s*s;
   double s3 = s2*s;
  
 /* FIXME: there are quite a few int's withouth a . in this file */
 //FP: eta2, eta3 can be avoided
 return eta*(3.4641016151377544 - 4.399247300629289*eta +
       9.397292189321194*eta2 - 13.180949901606242*eta3 +
       s*((1.0/eta - 0.0850917821418767 - 5.837029316602263*eta) +
       (0.1014665242971878 - 2.0967746996832157*eta)*s +
       (-1.3546806617824356 + 4.108962025369336*eta)*s2 +
       (-0.8676969352555539 + 2.064046835273906*eta)*s3));
 }
  
 /**
  * Wrapper function for FinalSpin0815_s.
  */
 static double FinalSpin0815(double eta, double chi1, double chi2) {
   // Convention m1 >= m2
   double Seta = sqrt(1.0 - 4.0*eta);
   double m1 = 0.5 * (1.0 + Seta);
   double m2 = 0.5 * (1.0 - Seta);
   double m1s = m1*m1;
   double m2s = m2*m2;
   // s defined around Equation 3.6 arXiv:1508.07250
   double s = (m1s * chi1 + m2s * chi2);
   return FinalSpin0815_s(eta, s);
 }
 static REAL8 FinalSpinIMRPhenomD_all_in_plane_spin_on_larger_BH(
   const REAL8 m1,     /**< Mass of companion 1 (solar masses) */
   const REAL8 m2,     /**< Mass of companion 2 (solar masses) */
   const REAL8 chi1_l, /**< Aligned spin of BH 1 */
   const REAL8 chi2_l, /**< Aligned spin of BH 2 */
   const REAL8 chip)   /**< Dimensionless spin in the orbital plane */
 {
   const REAL8 M = m1+m2;
   REAL8 eta = m1*m2/(M*M);
   //if (eta > 0.25) nudge(&eta, 0.25, 1e-6);
  
   REAL8 af_parallel, q_factor;
   if (m1 >= m2) {
     q_factor = m1/M;
     af_parallel = FinalSpin0815(eta, chi1_l, chi2_l);
   }
   else {
     q_factor = m2/M;
     af_parallel = FinalSpin0815(eta, chi2_l, chi1_l);
   }
  
   REAL8 Sperp = chip * q_factor*q_factor;
   REAL8 af = copysign(1.0, af_parallel) * sqrt(Sperp*Sperp + af_parallel*af_parallel);
   return af;
 }

 typedef char CHAR;  
  typedef uint32_t UINT4;         /**< Four-byte unsigned integer. */
  /** Length of name fields of LAL structured data types. */
 enum enumLALNameLength { LALNameLength = 64 };

 typedef struct tagREAL8Vector {
 #ifdef SWIG     /* SWIG interface directives */
     SWIGLAL(ARRAY_STRUCT_1D(REAL8Vector, REAL8, data, UINT4, length));
 #endif /* SWIG */
     UINT4 length; /**< Number of elements in array. */
     REAL8 *data; /**< Pointer to the data array. */
 } REAL8Vector;

typedef REAL8Vector REAL8Sequence;  

 /** Vector of type COMPLEX16, see \ref ss_Vector for more details. */
 typedef struct tagCOMPLEX16Vector {
 #ifdef SWIG     /* SWIG interface directives */
     SWIGLAL(ARRAY_STRUCT_1D(COMPLEX16Vector, COMPLEX16, data, UINT4, length));
 #endif /* SWIG */
     UINT4 length;    /**< Number of elements in array. */
     COMPLEX16 *data; /**< Pointer to the data array. */
 } COMPLEX16Vector;
  
typedef COMPLEX16Vector COMPLEX16Sequence;


 /** See \ref ss_FrequencySeries for documentation */
 typedef struct tagCOMPLEX16FrequencySeries {
     CHAR name[LALNameLength];
     REAL8 f0;
     REAL8 deltaF;
     COMPLEX16Sequence *data;
 } COMPLEX16FrequencySeries;

 typedef enum tagNRTidal_version_type {
  NRTidal_V, /**< version NRTidal: based on https://arxiv.org/pdf/1706.02969.pdf*/
  NRTidalv2_V, /**< version NRTidalv2: https://arxiv.org/abs/1905.06011 */
  NRTidalv2NoAmpCorr_V, /**< version NRTidalv2, without amplitude corrections */
  NRTidalv2NSBH_V, /**< version NRTidalv2: https://arxiv.org/abs/1905.06011 with amplitude corrections for NSBH (used for SEOBNRv4ROM_NRTidalv2_NSBH) */
  NoNRT_V /**< special case for PhenomPv2 BBH baseline */
 } NRTidal_version_type;
  
/*********************************************************************/
 /* This structure stores the PN coefficients used to calculate flux  */
 /* and waveform amplitude, and Fourier phase. It also stores some    */
 /* frequently used expressions which are constant during waveform    */
 /* generation.                                                       */
 /*********************************************************************/
  
 // MP: could we move this into the header file?
 typedef struct tagBBHPhenomCParams{
   REAL8 piM;
   REAL8 m_sec;
  
   REAL8 fmin;
   REAL8 fCut;
   REAL8 df;
  
   REAL8 f0;
   REAL8 f1;
   REAL8 f2;
   REAL8 d0;
   REAL8 d1;
   REAL8 d2;
  
   REAL8 afin;
   REAL8 fRingDown;
   REAL8 MfRingDown;
   REAL8 Qual;
  
   REAL8 pfaN;
   REAL8 pfa1;
   REAL8 pfa2;
   REAL8 pfa3;
   REAL8 pfa4;
   REAL8 pfa5;
   REAL8 pfa6;
   REAL8 pfa6log;
   REAL8 pfa7;
  
   REAL8 xdotaN;
   REAL8 xdota2;
   REAL8 xdota3;
   REAL8 xdota4;
   REAL8 xdota5;
   REAL8 xdota6;
   REAL8 xdota6log;
   REAL8 xdota7;
  
   REAL8 AN;
   REAL8 A2;
   REAL8 A3;
   REAL8 A4;
   REAL8 A5;
   REAL8 A5imag;
   REAL8 A6;
   REAL8 A6log;
   REAL8 A6imag;
  
   REAL8 a1;
   REAL8 a2;
   REAL8 a3;
   REAL8 a4;
   REAL8 a5;
   REAL8 a6;
   REAL8 g1;
   REAL8 del1;
   REAL8 del2;
   REAL8 b1;
   REAL8 b2;
 }
 BBHPhenomCParams;

 /**
  * Structure for passing around PN phasing coefficients.
  * For use with the TaylorF2 waveform.
  */
 #define PN_PHASING_SERIES_MAX_ORDER 15
 typedef struct tagPNPhasingSeries
 {
     REAL8 v[PN_PHASING_SERIES_MAX_ORDER+1];
     REAL8 vlogv[PN_PHASING_SERIES_MAX_ORDER+1];
     REAL8 vlogvsq[PN_PHASING_SERIES_MAX_ORDER+1];
     REAL8 vneg[PN_PHASING_SERIES_MAX_ORDER+1];
 }
 PNPhasingSeries;


/**
  * Internal core function to calculate
  * plus and cross polarizations of the PhenomP model
  * for a set of frequencies.
  * This can handle either user-specified frequency points
  * or create an equally-spaced frequency series.
  */
 static int PhenomPCore(
   COMPLEX16FrequencySeries **hptilde,        /**< [out] Frequency-domain waveform h+ */
   COMPLEX16FrequencySeries **hctilde,        /**< [out] Frequency-domain waveform hx */
   const REAL8 chi1_l_in,                     /**< Dimensionless aligned spin on companion 1 */
   const REAL8 chi2_l_in,                     /**< Dimensionless aligned spin on companion 2 */
   const REAL8 chip,                          /**< Effective spin in the orbital plane */
   const REAL8 thetaJ,                        /**< Angle between J0 and line of sight (z-direction) */
   const REAL8 m1_SI_in,                      /**< Mass of companion 1 (kg) */
   const REAL8 m2_SI_in,                      /**< Mass of companion 2 (kg) */
   const REAL8 distance,                      /**< Distance of source (m) */
   const REAL8 alpha0,                        /**< Initial value of alpha angle (azimuthal precession angle) */
   const REAL8 phic,                          /**< Orbital phase at the peak of the underlying non precessing model (rad) */
   const REAL8 f_ref,                         /**< Reference frequency */
   const REAL8Sequence *freqs_in,             /**< Frequency points at which to evaluate the waveform (Hz) */
   double deltaF,                             /**< Sampling frequency (Hz).
    * If deltaF > 0, the frequency points given in freqs are uniformly spaced with
    * spacing deltaF. Otherwise, the frequency points are spaced non-uniformly.
    * Then we will use deltaF = 0 to create the frequency series we return. */
   IMRPhenomP_version_type IMRPhenomP_version, /**< IMRPhenomPv1 uses IMRPhenomC, IMRPhenomPv2 uses IMRPhenomD, IMRPhenomPv2_NRTidal uses NRTidal framework with IMRPhenomPv2 */
   NRTidal_version_type NRTidal_version /**< either NRTidal or NRTidalv2 for BNS waveform; NoNRT_V for BBH waveform */
   //LALDict *extraParams /**<linked list that may contain the extra testing GR parameters and/or tidal parameters */
   )
 {
   /* Check inputs for sanity */
   //XLAL_CHECK(NULL != hptilde, XLAL_EFAULT);
   //XLAL_CHECK(NULL != hctilde, XLAL_EFAULT);
   //XLAL_CHECK(*hptilde == NULL, XLAL_EFAULT);
   //XLAL_CHECK(*hctilde == NULL, XLAL_EFAULT);
   //XLAL_CHECK(deltaF >= 0, XLAL_EDOM, "deltaF must be non-negative.\n");
   //XLAL_CHECK(m1_SI_in > 0, XLAL_EDOM, "m1 must be positive.\n");
   //XLAL_CHECK(m2_SI_in > 0, XLAL_EDOM, "m2 must be positive.\n");
   //XLAL_CHECK(f_ref > 0, XLAL_EDOM, "Reference frequency must be non-negative.\n");
   //XLAL_CHECK(distance > 0, XLAL_EDOM, "distance must be positive.\n");
   //XLAL_CHECK(fabs(chi1_l_in) <= 1.0, XLAL_EDOM, "Aligned spin chi1_l=%g must be <= 1 in magnitude!\n", chi1_l_in);
   //XLAL_CHECK(fabs(chi2_l_in) <= 1.0, XLAL_EDOM, "Aligned spin chi2_l=%g must be <= 1 in magnitude!\n", chi2_l_in);
   //XLAL_CHECK(fabs(chip) <= 1.0, XLAL_EDOM, "In-plane spin chip =%g must be <= 1 in magnitude!\n", chip);
  
   // See Fig. 1. in arxiv:1408.1810 for diagram of the angles.
   // Note that the angles phiJ which is calculated internally in XLALSimIMRPhenomPCalculateModelParametersFromSourceFrame
   // and alpha0 are degenerate. Therefore phiJ is not passed to this function.
   /* Phenomenological parameters */
   IMRPhenomDAmplitudeCoefficients *pAmp = NULL;
   IMRPhenomDPhaseCoefficients *pPhi = NULL;
   BBHPhenomCParams *PCparams = NULL;
   PNPhasingSeries *pn = NULL;
   // Spline
   gsl_interp_accel *acc_fixed = NULL;
   gsl_spline *phiI_fixed = NULL;
   REAL8Sequence *freqs_fixed = NULL;
   REAL8Sequence *phase_fixed = NULL;
   REAL8Sequence *freqs = NULL;
   int errcode = XLAL_SUCCESS;
   //LALDict *extraParams_in = extraParams;
   // Tidal corrections
   REAL8Sequence *phi_tidal = NULL;
   REAL8Sequence *amp_tidal = NULL;
   REAL8Sequence *planck_taper = NULL;
   REAL8Sequence *phi_tidal_fixed = NULL;
   REAL8Sequence *amp_tidal_fixed = NULL;
   REAL8Sequence *planck_taper_fixed = NULL;
   int ret = 0;
  
   // Enforce convention m2 >= m1
   REAL8 chi1_l, chi2_l;
   REAL8 m1_SI, m2_SI;
   REAL8 lambda1_in = 0.0;
   REAL8 lambda2_in = 0.0;
   REAL8 quadparam1_in = 1.0;
   REAL8 quadparam2_in = 1.0;
  
  // if (IMRPhenomP_version == IMRPhenomPv2NRTidal_V) {
  //   int retcode;
  //   retcode = XLALSimInspiralSetQuadMonParamsFromLambdas(extraParams);
  //   XLAL_CHECK(retcode == XLAL_SUCCESS, XLAL_EFUNC, "Failed to set quadparams from Universal relation.\n");
  //   lambda1_in = XLALSimInspiralWaveformParamsLookupTidalLambda1(extraParams);
  //   lambda2_in = XLALSimInspiralWaveformParamsLookupTidalLambda2(extraParams);
  //   quadparam1_in = 1. + XLALSimInspiralWaveformParamsLookupdQuadMon1(extraParams);
  //   quadparam2_in = 1. + XLALSimInspiralWaveformParamsLookupdQuadMon2(extraParams);
  // }
  
   REAL8 lambda1, lambda2;
   REAL8 quadparam1, quadparam2;
   /* declare HO 3.5PN spin-spin and spin-cubed terms added later to Pv2_NRTidalv2 */
   REAL8 SS_3p5PN = 0., SSS_3p5PN = 0.;
   REAL8 SS_3p5PN_n = 0., SSS_3p5PN_n = 0.;
  
   if (m2_SI_in >= m1_SI_in) {
     m1_SI = m1_SI_in;
     m2_SI = m2_SI_in;
     chi1_l = chi1_l_in;
     chi2_l = chi2_l_in;
     lambda1 = lambda1_in;
     lambda2 = lambda2_in;
     quadparam1 = quadparam1_in;
     quadparam2 = quadparam2_in;
   }
   else { // swap bodies 1 <-> 2
     m1_SI = m2_SI_in;
     m2_SI = m1_SI_in;
     chi1_l = chi2_l_in;
     chi2_l = chi1_l_in;
     lambda1 = lambda2_in;
     lambda2 = lambda1_in;
     quadparam1 = quadparam2_in;
     quadparam2 = quadparam1_in;
   }
  
   errcode = init_useful_powers(&powers_of_pi, LAL_PI);
  
   /* Find frequency bounds */
   if (!freqs_in || !freqs_in->data) XLAL_ERROR(XLAL_EFAULT);
   double f_min = freqs_in->data[0];
   double f_max = freqs_in->data[freqs_in->length - 1];

  
   /* External units: SI; internal units: solar masses */
   const REAL8 m1 = m1_SI / LAL_MSUN_SI;
   const REAL8 m2 = m2_SI / LAL_MSUN_SI;
   const REAL8 M = m1 + m2;
   const REAL8 m_sec = M * LAL_MTSUN_SI;   /* Total mass in seconds */
   REAL8 q = m2 / m1; /* q >= 1 */
   REAL8 eta = m1 * m2 / (M*M);    /* Symmetric mass-ratio */
   const REAL8 piM = LAL_PI * m_sec;
   /* New variables needed for the NRTidalv2 model */
   REAL8 X_A = m1/M;
   REAL8 X_B = m2/M;
   REAL8 pn_fac = 3.*pow(piM,2./3.)/(128.*eta);
  
   LIGOTimeGPS ligotimegps_zero = LIGOTIMEGPSZERO; // = {0, 0}
  
   // Note:
   // * IMRPhenomP uses chi_eff both in the aligned part and the twisting
   // * IMRPhenomPv2 uses chi1_l, chi2_l in the aligned part and chi_eff in the twisting
   const REAL8 chi_eff = (m1*chi1_l + m2*chi2_l) / M; /* Effective aligned spin */
   const REAL8 chil = (1.0+q)/q * chi_eff; /* dimensionless aligned spin of the largest BH */
  
   if (eta > 0.25 || q < 1.0) {
     nudge(&eta, 0.25, 1e-6);
     nudge(&q, 1.0, 1e-6);
   }
  
   NNLOanglecoeffs angcoeffs; /* Next-to-next-to leading order PN coefficients for Euler angles alpha and epsilon */
   ComputeNNLOanglecoeffs(&angcoeffs,q,chil,chip);
  
   /* Compute the offsets due to the choice of integration constant in alpha and epsilon PN formula */
   const REAL8 omega_ref = piM * f_ref;
   const REAL8 logomega_ref = log(omega_ref);
   const REAL8 omega_ref_cbrt = cbrt(piM * f_ref); // == v0
   const REAL8 omega_ref_cbrt2 = omega_ref_cbrt*omega_ref_cbrt;
   const REAL8 alphaNNLOoffset = (angcoeffs.alphacoeff1/omega_ref
                               + angcoeffs.alphacoeff2/omega_ref_cbrt2
                               + angcoeffs.alphacoeff3/omega_ref_cbrt
                               + angcoeffs.alphacoeff4*logomega_ref
                               + angcoeffs.alphacoeff5*omega_ref_cbrt);
  
   const REAL8 epsilonNNLOoffset = (angcoeffs.epsiloncoeff1/omega_ref
                                 + angcoeffs.epsiloncoeff2/omega_ref_cbrt2
                                 + angcoeffs.epsiloncoeff3/omega_ref_cbrt
                                 + angcoeffs.epsiloncoeff4*logomega_ref
                                 + angcoeffs.epsiloncoeff5*omega_ref_cbrt);
  
   /* Compute Ylm's only once and pass them to PhenomPCoreOneFrequency() below. */
   SpinWeightedSphericalHarmonic_l2 Y2m;
   const REAL8 ytheta  = thetaJ;
   const REAL8 yphi    = 0;
   Y2m.Y2m2 = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2, -2);
   Y2m.Y2m1 = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2, -1);
   Y2m.Y20  = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2,  0);
   Y2m.Y21  = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2,  1);
   Y2m.Y22  = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2,  2);
  
  
   REAL8 fCut = 0.0;
   REAL8 finspin = 0.0;
   REAL8 f_final = 0.0;
  
   switch (IMRPhenomP_version) {
     //case IMRPhenomPv1_V:
     //  XLAL_PRINT_INFO("*** IMRPhenomP version 1: based on IMRPhenomC ***");
       // PhenomC with ringdown using Barausse 2009 formula for final spin
     //  PCparams = ComputeIMRPhenomCParamsRDmod(m1, m2, chi_eff, chip, extraParams);
      // if (!PCparams) {
      //   errcode = XLAL_EFUNC;
      //   goto cleanup;
      // }
     //  fCut = PCparams->fCut;
     //  f_final = PCparams->fRingDown;
       break;
     case IMRPhenomPv2_V:
     case IMRPhenomPv2NRTidal_V:
       XLAL_PRINT_INFO("*** IMRPhenomP version 2: based on IMRPhenomD ***");
       // PhenomD uses FinalSpin0815() to calculate the final spin if the spins are aligned.
       // We use a generalized version of FinalSpin0815() that includes the in-plane spin chip.
       finspin = FinalSpinIMRPhenomD_all_in_plane_spin_on_larger_BH(m1, m2, chi1_l, chi2_l, chip);
       if( fabs(finspin) > 1.0 ) {
         XLAL_PRINT_WARNING("Warning: final spin magnitude %g > 1. Setting final spin magnitude = 1.", finspin);
         finspin = copysign(1.0, finspin);
       }
       // IMRPhenomD assumes that m1 >= m2.
       pAmp = XLALMalloc(sizeof(IMRPhenomDAmplitudeCoefficients));
       ComputeIMRPhenomDAmplitudeCoefficients(pAmp, eta, chi2_l, chi1_l, finspin);
       pPhi = XLALMalloc(sizeof(IMRPhenomDPhaseCoefficients));
       ComputeIMRPhenomDPhaseCoefficients(pPhi, eta, chi2_l, chi1_l, finspin);
       if (extraParams==NULL)
       {
               extraParams=XLALCreateDict();
       }
       XLALSimInspiralWaveformParamsInsertPNSpinOrder(extraParams, LAL_SIM_INSPIRAL_SPIN_ORDER_35PN);
       // Start making changes here: use XLALSimInspiralWaveformParamsInsertdQuadMon1() function
       //if (IMRPhenomP_version == IMRPhenomPv2NRTidal_V) {
       //  XLALSimInspiralWaveformParamsInsertTidalLambda1(extraParams, lambda1);
       //  XLALSimInspiralWaveformParamsInsertTidalLambda2(extraParams, lambda2);
       //  XLALSimInspiralWaveformParamsInsertdQuadMon1(extraParams, quadparam1-1.);
       //  XLALSimInspiralWaveformParamsInsertdQuadMon2(extraParams, quadparam2-1.);
       //  XLALSimInspiralTaylorF2AlignedPhasing(&pn, m1, m2, chi1_l, chi2_l, extraParams);
       //} else {
         XLALSimInspiralTaylorF2AlignedPhasing(&pn, m1, m2, chi1_l, chi2_l, extraParams);
       //}
  
       if (!pAmp || !pPhi || !pn) {
         errcode = XLAL_EFUNC;
         goto cleanup;
       }
  
       // Subtract 3PN spin-spin term below as this is in LAL's TaylorF2 implementation
       // (LALSimInspiralPNCoefficients.c -> XLALSimInspiralPNPhasing_F2), but
       // was not available when PhenomD was tuned.
       pn->v[6] -= (Subtract3PNSS(m1, m2, M, eta, chi1_l, chi2_l) * pn->v[0]);
  
       PhiInsPrefactors phi_prefactors;
       errcode = init_phi_ins_prefactors(&phi_prefactors, pPhi, pn);
       XLAL_CHECK(XLAL_SUCCESS == errcode, errcode, "init_phi_ins_prefactors failed");
  
       ComputeIMRPhenDPhaseConnectionCoefficients(pPhi, pn, &phi_prefactors, 1.0, 1.0);
       // This should be the same as the ending frequency in PhenomD
       fCut = f_CUT / m_sec;
       f_final = pAmp->fRD / m_sec;
       break;
     default:
       XLALPrintError( "XLAL Error - %s: Unknown IMRPhenomP version!\nAt present only v1 and v2 are available.\n", __func__);
       errcode = XLAL_EINVAL;
       goto cleanup;
       break;
   }
  
   XLAL_CHECK ( fCut > f_min, XLAL_EDOM, "fCut = %.2g/M <= f_min", fCut );
  
   /* Default f_max to params->fCut */
   REAL8 f_max_prime = f_max ? f_max : fCut;
   f_max_prime = (f_max_prime > fCut) ? fCut : f_max_prime;
   if (f_max_prime <= f_min){
     XLALPrintError("XLAL Error - %s: f_max <= f_min\n", __func__);
     errcode = XLAL_EDOM;
     goto cleanup;
   }
  
   /* Allocate hp, hc */
  
   UINT4 L_fCut = 0; // number of frequency points before we hit fCut
   size_t n = 0;
   UINT4 offset = 0; // Index shift between freqs and the frequency series
   if (deltaF > 0)  { // freqs contains uniform frequency grid with spacing deltaF; we start at frequency 0
     /* Set up output array with size closest power of 2 */
     if (f_max_prime < f_max)  /* Resize waveform if user wants f_max larger than cutoff frequency */
       n = NextPow2(f_max / deltaF) + 1;
     else
       n = NextPow2(f_max_prime / deltaF) + 1;
  
     /* coalesce at t=0 */
     XLAL_CHECK(XLALGPSAdd(&ligotimegps_zero, -1. / deltaF), XLAL_EFUNC,
     "Failed to shift coalescence time by -1.0/deltaF with deltaF=%g.", deltaF); // shift by overall length in time
  
     *hptilde = XLALCreateCOMPLEX16FrequencySeries("hptilde: FD waveform", &ligotimegps_zero, 0.0, deltaF, &lalStrainUnit, n);
     if(!*hptilde) {
       errcode = XLAL_ENOMEM;
       goto cleanup;
     }
     *hctilde = XLALCreateCOMPLEX16FrequencySeries("hctilde: FD waveform", &ligotimegps_zero, 0.0, deltaF, &lalStrainUnit, n);
     if(!*hctilde) {
       errcode = XLAL_ENOMEM;
       goto cleanup;
     }
  
     // Recreate freqs using only the lower and upper bounds
     size_t i_min = (size_t) (f_min / deltaF);
     size_t i_max = (size_t) (f_max_prime / deltaF);
     freqs = XLALCreateREAL8Sequence(i_max - i_min);
     if (!freqs) {
       errcode = XLAL_EFUNC;
       XLALPrintError("XLAL Error - %s: Frequency array allocation failed.", __func__);
       goto cleanup;
     }
     for (UINT4 i=i_min; i<i_max; i++)
       freqs->data[i-i_min] = i*deltaF;
     L_fCut = freqs->length;
     offset = i_min;
   } else { // freqs contains frequencies with non-uniform spacing; we start at lowest given frequency
     n = freqs_in->length;
  
     *hptilde = XLALCreateCOMPLEX16FrequencySeries("hptilde: FD waveform", &ligotimegps_zero, f_min, 0, &lalStrainUnit, n);
     if(!*hptilde) {
       XLALPrintError("XLAL Error - %s: Failed to allocate frequency series for hptilde polarization with f_min=%f and %tu bins.", __func__, f_min, n);
       errcode = XLAL_ENOMEM;
       goto cleanup;
     }
     *hctilde = XLALCreateCOMPLEX16FrequencySeries("hctilde: FD waveform", &ligotimegps_zero, f_min, 0, &lalStrainUnit, n);
     if(!*hctilde) {
       XLALPrintError("XLAL Error - %s: Failed to allocate frequency series for hctilde polarization with f_min=%f and %tu bins.", __func__, f_min, n);
       errcode = XLAL_ENOMEM;
       goto cleanup;
     }
     offset = 0;
  
     // Enforce that FS is strictly increasing
     // (This is needed for phase correction towards the end of this function.)
     for (UINT4 i=1; i<n; i++)
     {
       if (!(freqs_in->data[i] > freqs_in->data[i-1]))
       {
         XLALPrintError("XLAL Error - %s: Frequency sequence must be strictly increasing!\n",  __func__);
         errcode = XLAL_EDOM;
         goto cleanup;
       }
     }
  
     freqs = XLALCreateREAL8Sequence(n);
     if (!freqs) {
       XLALPrintError("XLAL Error - %s: Frequency array allocation failed.",  __func__);
       errcode = XLAL_ENOMEM;
       goto cleanup;
     }
     // Restrict sequence to frequencies <= fCut
     for (UINT4 i=0; i<n; i++)
       if (freqs_in->data[i] <= fCut) {
         freqs->data[i] = freqs_in->data[i];
         L_fCut++;
       }
   }
  
  
   memset((*hptilde)->data->data, 0, n * sizeof(COMPLEX16));
   memset((*hctilde)->data->data, 0, n * sizeof(COMPLEX16));
   XLALUnitMultiply(&((*hptilde)->sampleUnits), &((*hptilde)->sampleUnits), &lalSecondUnit);
   XLALUnitMultiply(&((*hctilde)->sampleUnits), &((*hctilde)->sampleUnits), &lalSecondUnit);
  
   if (IMRPhenomP_version == IMRPhenomPv2NRTidal_V) {
     /* Generating the NR tidal amplitude and phase */
     /* Get FD tidal phase correction and amplitude factor from arXiv:1706.02969 */
     if (NRTidal_version == NRTidal_V) {
       phi_tidal = XLALCreateREAL8Sequence(L_fCut);
       planck_taper = XLALCreateREAL8Sequence(L_fCut);
       ret = XLALSimNRTunedTidesFDTidalPhaseFrequencySeries(phi_tidal, amp_tidal, planck_taper, freqs, m1_SI, m2_SI, lambda1, lambda2, NRTidal_V);
       XLAL_CHECK(XLAL_SUCCESS == ret, ret, "XLALSimNRTunedTidesFDTidalPhaseFrequencySeries Failed.");
     }
     else if (NRTidal_version == NRTidalv2_V) {
       phi_tidal = XLALCreateREAL8Sequence(L_fCut);
       amp_tidal = XLALCreateREAL8Sequence(L_fCut);
       planck_taper = XLALCreateREAL8Sequence(L_fCut);
       ret = XLALSimNRTunedTidesFDTidalPhaseFrequencySeries(phi_tidal, amp_tidal, planck_taper, freqs, m1_SI, m2_SI, lambda1, lambda2, NRTidalv2_V);
       XLAL_CHECK(XLAL_SUCCESS == ret, ret, "XLALSimNRTunedTidesFDTidalPhaseFrequencySeries Failed.");
       /* Get the PN SS-tail and SSS terms */
       XLALSimInspiralGetHOSpinTerms(&SS_3p5PN, &SSS_3p5PN, X_A, X_B, chi1_l, chi2_l, quadparam1, quadparam2);
     }
   }
  
 //   phis = XLALMalloc(L_fCut*sizeof(REAL8)); // array for waveform phase
 //   if(!phis) {
 //     errcode = XLAL_ENOMEM;
 //     goto cleanup;
 //   }
  
   AmpInsPrefactors amp_prefactors;
   PhiInsPrefactors phi_prefactors;
  
   if (IMRPhenomP_version == IMRPhenomPv2_V || IMRPhenomP_version == IMRPhenomPv2NRTidal_V) {
     errcode = init_amp_ins_prefactors(&amp_prefactors, pAmp);
     XLAL_CHECK(XLAL_SUCCESS == errcode, errcode, "init_amp_ins_prefactors() failed.");
     errcode = init_phi_ins_prefactors(&phi_prefactors, pPhi, pn);
     XLAL_CHECK(XLAL_SUCCESS == errcode, errcode, "init_phi_ins_prefactors() failed.");
   }
  
   /*
     We can't call XLAL_ERROR() directly with OpenMP on.
     Keep track of return codes for each thread and in addition use flush to get out of
     the parallel for loop as soon as possible if something went wrong in any thread.
   */
   #pragma omp parallel for
   for (UINT4 i=0; i<L_fCut; i++) { // loop over frequency points in sequence
     COMPLEX16 hPhenom = 0.0; // IMRPhenom waveform (before precession) at a given frequency point
     COMPLEX16 hp_val = 0.0;
     COMPLEX16 hc_val = 0.0;
     REAL8 phasing = 0;
     double f = freqs->data[i];
     int j = i + offset; // shift index for frequency series if needed
  
     int per_thread_errcode=0;
  
     #pragma omp flush(errcode)
     if (errcode != XLAL_SUCCESS)
       goto skip;
  
     /* Generate the waveform */
     if (IMRPhenomP_version == IMRPhenomPv2NRTidal_V) {
       if (NRTidal_version == NRTidal_V) {
         double window = planck_taper->data[i];
         REAL8 phaseTidal = phi_tidal->data[i];
         per_thread_errcode = PhenomPCoreOneFrequency_withTides(f, window, phaseTidal, 0.0, distance, M, phic,
                               pAmp, pPhi, pn,
                               &hPhenom, &phasing, &amp_prefactors, &phi_prefactors);
       }
        else if (NRTidal_version == NRTidalv2_V) {
          double ampTidal = amp_tidal->data[i];
          double window = planck_taper->data[i];
  
         /* 
          Compute the tidal phase correction and add the 3.5PN SS and SSS contributions which 
          are not incorporated in the TaylorF2 baseline
         */
          REAL8 phaseTidal =  phi_tidal->data[i] + pn_fac*(SS_3p5PN + SSS_3p5PN)*pow(f,2./3.);
  
          per_thread_errcode = PhenomPCoreOneFrequency_withTides(f, window, phaseTidal, ampTidal, distance, M, phic,
                               pAmp, pPhi, pn,
                               &hPhenom, &phasing, &amp_prefactors, &phi_prefactors);
          }
       } else {
       per_thread_errcode = PhenomPCoreOneFrequency(f, eta, distance, M, phic,
                               pAmp, pPhi, PCparams, pn,
                               &hPhenom, &phasing, IMRPhenomP_version, &amp_prefactors, &phi_prefactors);
      }
  
     if (per_thread_errcode != XLAL_SUCCESS) {
       errcode = per_thread_errcode;
      }
  
     per_thread_errcode = PhenomPCoreTwistUp(f, hPhenom, eta, chi1_l, chi2_l, chip, M,
                               &angcoeffs, &Y2m,
                               alphaNNLOoffset - alpha0, epsilonNNLOoffset,
                               &hp_val, &hc_val, IMRPhenomP_version);
  
     if (per_thread_errcode != XLAL_SUCCESS) {
       errcode = per_thread_errcode;
       #pragma omp flush(errcode)
     }
  
     ((*hptilde)->data->data)[j] = hp_val;
     ((*hctilde)->data->data)[j] = hc_val;
  
 //    phis[i] = phasing;
  
     skip: /* this statement intentionally left blank */;
   }
  
  
   /* The next part computes and applies a time shift correction to make the waveform peak at t=0 */
  
   /* Set up spline for phase on fixed grid */
   const int n_fixed = 10;
   freqs_fixed = XLALCreateREAL8Sequence(n_fixed);
   XLAL_CHECK(freqs_fixed != NULL, XLAL_EFAULT);
   phase_fixed = XLALCreateREAL8Sequence(n_fixed);
   XLAL_CHECK(phase_fixed != NULL, XLAL_EFAULT);
  
   /* For BNS waveforms, ending frequency is f_merger; putting f_final to f_merger for IMRPhenomPv2_NRTidal and IMRPhenomPv2_NRTidalv2  */
   if (IMRPhenomP_version == IMRPhenomPv2NRTidal_V) {
     REAL8 kappa2T = XLALSimNRTunedTidesComputeKappa2T(m1_SI, m2_SI, lambda1, lambda2);
     REAL8 f_merger = XLALSimNRTunedTidesMergerFrequency(M, kappa2T, q);
     f_final = f_merger;
   }
  
   /* Set up fixed frequency grid around ringdown frequency for BBH (IMRPhenomPv2) waveforms,
      for IMRPhenomPv2_NRTidal and IMRPhenomPv2_NRTidalv2 waveforms, the grid is set up around the merger frequency */
   REAL8 freqs_fixed_start = 0.8*f_final;
   REAL8 freqs_fixed_stop = 1.2*f_final;
   if (freqs_fixed_stop > fCut)
       freqs_fixed_stop = fCut;
   REAL8 delta_freqs_fixed = (freqs_fixed_stop - freqs_fixed_start) / (n_fixed - 1);
   for (int i=0; i < n_fixed; i++)
     freqs_fixed->data[i] = freqs_fixed_start + i*delta_freqs_fixed;
  
   /* Recompute tidal corrections if needed */
   if (IMRPhenomP_version == IMRPhenomPv2NRTidal_V) {
     /* Generating the NR tidal amplitude and phase for the fixed grid */
     /* Get FD tidal phase correction and amplitude factor from arXiv:1706.02969 */
     phi_tidal_fixed = XLALCreateREAL8Sequence(n_fixed);
     amp_tidal_fixed = XLALCreateREAL8Sequence(n_fixed);
     planck_taper_fixed = XLALCreateREAL8Sequence(n_fixed);
     ret = XLALSimNRTunedTidesFDTidalPhaseFrequencySeries(phi_tidal_fixed, amp_tidal_fixed, planck_taper_fixed,
                                                          freqs_fixed, m1_SI, m2_SI, lambda1, lambda2, NRTidal_version);
     XLAL_CHECK(XLAL_SUCCESS == ret, ret, "XLALSimNRTunedTidesFDTidalPhaseFrequencySeries Failed.");
     if (NRTidal_version == NRTidalv2_V)
       XLALSimInspiralGetHOSpinTerms(&SS_3p5PN_n, &SSS_3p5PN_n, X_A, X_B, chi1_l, chi2_l, quadparam1, quadparam2);
   }
  
   /* We need another loop to generate the phase values on the fixed grid; no need for OpenMP here */
   for (int i=0; i<n_fixed; i++) { // loop over frequency points in sequence
     COMPLEX16 hPhenom = 0.0; // IMRPhenom waveform (before precession) at a given frequency point
     REAL8 phasing = 0;
     double f = freqs_fixed->data[i];
     int per_thread_errcode = 0;
  
     if (errcode != XLAL_SUCCESS)
       goto skip_fixed;
  
     /* Generate the waveform */
     if (IMRPhenomP_version == IMRPhenomPv2NRTidal_V) {
       if (NRTidal_version == NRTidal_V) {
         
         double window = planck_taper_fixed->data[i];
         REAL8 phaseTidal = phi_tidal_fixed->data[i];
         per_thread_errcode = PhenomPCoreOneFrequency_withTides(f, window, phaseTidal, 0.0, distance, M, phic,
                               pAmp, pPhi, pn,
                               &hPhenom, &phasing, &amp_prefactors, &phi_prefactors);
         }
       else if (NRTidal_version == NRTidalv2_V) {
         double ampTidal = amp_tidal_fixed->data[i];
         double window = planck_taper_fixed->data[i];
         REAL8 phaseTidal =  phi_tidal_fixed->data[i] + pn_fac*(SS_3p5PN_n + SSS_3p5PN_n)*pow(f,2./3.);
         per_thread_errcode = PhenomPCoreOneFrequency_withTides(f, window, phaseTidal, ampTidal, distance, M, phic,
                               pAmp, pPhi, pn,
                               &hPhenom, &phasing, &amp_prefactors, &phi_prefactors); 
                               
         }                        
       } else {
       per_thread_errcode = PhenomPCoreOneFrequency(f, eta, distance, M, phic,
                               pAmp, pPhi, PCparams, pn,
                               &hPhenom, &phasing, IMRPhenomP_version, &amp_prefactors, &phi_prefactors);
      }
  
     if (per_thread_errcode != XLAL_SUCCESS) {
       errcode = per_thread_errcode;
     }
  
     phase_fixed->data[i] = phasing;
  
     skip_fixed: /* this statement intentionally left blank */;
   }
  
   /* Correct phasing so we coalesce at t=0 (with the definition of the epoch=-1/deltaF above) */
   /* We apply the same time shift to hptilde and hctilde based on the overall phasing returned by PhenomPCoreOneFrequency */
  
   /* Set up spline for phase */
   acc_fixed = gsl_interp_accel_alloc();
   phiI_fixed = gsl_spline_alloc(gsl_interp_cspline, n_fixed);
   XLAL_CHECK(phiI_fixed, XLAL_ENOMEM, "Failed to allocate GSL spline with %d points for phase.", n_fixed);
   REAL8 t_corr_fixed = 0.0;
   gsl_spline_init(phiI_fixed, freqs_fixed->data, phase_fixed->data, n_fixed);
   /* Time correction is t(f_final) = 1/(2pi) dphi/df (f_final) */
   t_corr_fixed = gsl_spline_eval_deriv(phiI_fixed, f_final, acc_fixed) / (2*LAL_PI);
   // XLAL_PRINT_INFO("t_corr (fixed grid): %g\n", t_corr_fixed);
  
   /* Now correct phase */
   for (UINT4 i=0; i<L_fCut; i++) { // loop over frequency points in user-specified sequence
     double f = freqs->data[i];
     COMPLEX16 phase_corr = (cos(2*LAL_PI * f * t_corr_fixed) - I*sin(2*LAL_PI * f * t_corr_fixed));
     int j = i + offset; // shift index for frequency series if needed
     ((*hptilde)->data->data)[j] *= phase_corr;
     ((*hctilde)->data->data)[j] *= phase_corr;
   }
  
  
   cleanup:
   if (freqs_fixed) XLALDestroyREAL8Sequence(freqs_fixed);
   if (phase_fixed) XLALDestroyREAL8Sequence(phase_fixed);
   if (phiI_fixed) gsl_spline_free(phiI_fixed);
   if (acc_fixed) gsl_interp_accel_free(acc_fixed);
  
   if(PCparams) XLALFree(PCparams);
   if(pAmp) XLALFree(pAmp);
   if(pPhi) XLALFree(pPhi);
   if(pn) XLALFree(pn);
  
   /* If extraParams was allocated in this function and not passed in
    * we need to free it to prevent a leak */
   if(extraParams && !extraParams_in) XLALDestroyDict(extraParams);
  
   if(freqs) XLALDestroyREAL8Sequence(freqs);
  
   if (phi_tidal) XLALDestroyREAL8Sequence(phi_tidal);
   if (amp_tidal) XLALDestroyREAL8Sequence(amp_tidal);
   if (planck_taper) XLALDestroyREAL8Sequence(planck_taper);
   if (amp_tidal_fixed) XLALDestroyREAL8Sequence(amp_tidal_fixed);
   if (phi_tidal_fixed) XLALDestroyREAL8Sequence(phi_tidal_fixed);
   if (planck_taper_fixed) XLALDestroyREAL8Sequence(planck_taper_fixed);
  
   if( errcode != XLAL_SUCCESS ) {
     if(*hptilde) {
       XLALDestroyCOMPLEX16FrequencySeries(*hptilde);
       *hptilde=NULL;
     }
     if(*hctilde) {
       XLALDestroyCOMPLEX16FrequencySeries(*hctilde);
       *hctilde=NULL;
     }
     XLAL_ERROR(errcode);
   }
   else {
     return XLAL_SUCCESS;
   }
 }

int main(){
    REAL8 tmp1, tmp2;
    REAL8 chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz;
    REAL8 m1_SI, m2_SI, f_ref, phiRef, incl, s1x, s1y, s1z, s2x, s2y, s2z;
    // IMPORTANT: ripple's phenomP assumes m1 > m2, 
    // but all functions here assume m1 < m2.
    // So the parameters here needs to be reversed
    
    m1_SI = 6 * LAL_MSUN_SI;
    m2_SI = 10 * LAL_MSUN_SI;
    f_ref = 30;
    phiRef = 0.0;
    incl = LAL_PI/2.0;
    s1x = 0.5;
    s1y = -0.2;
    s1z = -0.5;
    s2x = 0.1;
    s2y = 0.6;
    s2z = 0.5;

    //IMRPhenomP_version_type IMRPhenomPv2_V;

    XLALSimIMRPhenomPCalculateModelParametersFromSourceFrame(&chi1_l,&chi2_l, &chip, &thetaJN, &alpha0, &phi_aligned, &zeta_polariz, m1_SI, m2_SI, f_ref, phiRef, incl, s1x, s1y, s1z, s2x, s2y, s2z, IMRPhenomPv2_V);
    

    NNLOanglecoeffs angcoeffs; /* Next-to-next-to leading order PN coefficients for Euler angles alpha and epsilon */
    const REAL8 m1 = m1_SI / LAL_MSUN_SI;
    const REAL8 m2 = m2_SI / LAL_MSUN_SI;
    const REAL8 M = m1 + m2;
    const REAL8 m_sec = M * LAL_MTSUN_SI;   /* Total mass in seconds */
    REAL8 q = m2 / m1; /* q >= 1 */
    const REAL8 chi_eff = (m1*chi1_l + m2*chi2_l) / M; /* Effective aligned spin */
    const REAL8 chil = (1.0+q)/q * chi_eff; /* dimensionless aligned spin of the largest BH */
    const REAL8 eta = m1 * m2 / (M*M);    /* Symmetric mass-ratio */
    ///////////////
    // Final spin calculation
    /////////////// 
    double finspin = FinalSpinIMRPhenomD_all_in_plane_spin_on_larger_BH(m1, m2, chi1_l, chi2_l, chip);
    printf("finspin: %.10f \n", finspin);
    
    ComputeNNLOanglecoeffs(&angcoeffs,q,chil,chip);
    //printf("Here are the output of LALtoPhenomP: \n");
    printf("%.10f, %.10f, %.10f, %.10f, %.10f, %.10f %.10f", chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz);
    printf("\n");  

    printf("\n Here are the outputs of NNLOanglecoeffs:  \n");
    printf("%.10f, %.10f, %.10f, %.10f, %.10f \n", 
    angcoeffs.alphacoeff1, angcoeffs.alphacoeff2, angcoeffs.alphacoeff3, 
    angcoeffs.alphacoeff4, angcoeffs.alphacoeff5);
    
    printf("%.10f, %.10f, %.10f, %.10f, %.10f \n", 
    angcoeffs.epsiloncoeff1, angcoeffs.epsiloncoeff2, angcoeffs.epsiloncoeff3, 
    angcoeffs.epsiloncoeff4, angcoeffs.epsiloncoeff5);

    REAL8 cos_beta_half, sin_beta_half;
    WignerdCoefficients(&cos_beta_half, &sin_beta_half, 0.34, 0.52, 0.44, 0.135);
    printf("WignerdCoeffs: \n");
    printf("%.10f, %.10f \n",cos_beta_half, sin_beta_half);

   const REAL8 piM = LAL_PI * m_sec;

  /* Compute the offsets due to the choice of integration constant in alpha and epsilon PN formula */
   const REAL8 omega_ref = piM * f_ref;
   const REAL8 logomega_ref = log(omega_ref);
   const REAL8 omega_ref_cbrt = cbrt(piM * f_ref); // == v0
   const REAL8 omega_ref_cbrt2 = omega_ref_cbrt*omega_ref_cbrt;
   const REAL8 alphaNNLOoffset = (angcoeffs.alphacoeff1/omega_ref
                               + angcoeffs.alphacoeff2/omega_ref_cbrt2
                               + angcoeffs.alphacoeff3/omega_ref_cbrt
                               + angcoeffs.alphacoeff4*logomega_ref
                               + angcoeffs.alphacoeff5*omega_ref_cbrt);
  
   const REAL8 epsilonNNLOoffset = (angcoeffs.epsiloncoeff1/omega_ref
                                 + angcoeffs.epsiloncoeff2/omega_ref_cbrt2
                                 + angcoeffs.epsiloncoeff3/omega_ref_cbrt
                                 + angcoeffs.epsiloncoeff4*logomega_ref
                                 + angcoeffs.epsiloncoeff5*omega_ref_cbrt);

    COMPLEX16 hp, hc;
    SpinWeightedSphericalHarmonic_l2 Y2m;
    const REAL8 ytheta  = thetaJN;
    const REAL8 yphi    = 0;
    printf("ythetaa: %.10f \n", ytheta);
   Y2m.Y2m2 = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2, -2);
   Y2m.Y2m1 = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2, -1);
   Y2m.Y20  = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2,  0);
   Y2m.Y21  = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2,  1);
   Y2m.Y22  = XLALSpinWeightedSphericalHarmonic(ytheta, yphi, -2, 2,  2);
   printf("Here are the -2Ylms: \n");
   printf("%.10f+%.10fi ", creal(Y2m.Y2m2),cimag(Y2m.Y2m2));
   printf("%.10f+%.10fi ", creal(Y2m.Y2m1),cimag(Y2m.Y2m1));
   printf("%.10f+%.10fi ", creal(Y2m.Y20),cimag(Y2m.Y20));
   printf("%.10f+%.10fi ", creal(Y2m.Y21),cimag(Y2m.Y21));
   printf("%.10f+%.10fi \n", creal(Y2m.Y22),cimag(Y2m.Y22));


   COMPLEX16 fake_hPhenom[4], fake_fHz[4];
   fake_hPhenom[0] = 1;
   fake_hPhenom[1] = 1+3I;
   fake_hPhenom[2] = 0.04-95I;
   fake_hPhenom[3] = -3.87-0.001I;
   fake_fHz[0] = 100;
   fake_fHz[1] = 25.0;
   fake_fHz[2] = 96.699;
   fake_fHz[3] = 238.75565;

   printf("offsets: %.10f %.10f \n", alphaNNLOoffset, epsilonNNLOoffset);
   printf("parameters of twistup: %.10f %.10f %.10f %.10f %.10f \n",
          eta, chi1_l, chi2_l, chip, M);
   for (int i=0; i < 4; i++){
       PhenomPCoreTwistUp(fake_fHz[i], fake_hPhenom[i], eta, chi1_l, chi2_l, chip, M,
                          &angcoeffs, &Y2m, alphaNNLOoffset-alpha0,
                          epsilonNNLOoffset, &hp, &hc,IMRPhenomPv2_V);
       printf("%d iteration hp and hc : %.10f + i%.10f, %.10f + i%.10f \n", 
            i, creal(hp), cimag(hp),creal(hc), cimag(hc));
   }
    


    return 0;
}