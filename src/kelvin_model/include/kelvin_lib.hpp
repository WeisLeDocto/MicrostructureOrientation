#ifdef _MSC_VER
    #define EXPORT_SYMBOL __declspec(dllexport)
#else
    #define EXPORT_SYMBOL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double exx;
    double eyy;
    double exy;
    double theta_1;
    double theta_2;
    double theta_3;
    double sigma_1;
    double sigma_2;
    double sigma_3;
    double density;
} PixelData;

typedef struct {
    double lambda_h;
    double lambda_11, lambda_21, lambda_31, lambda_41, lambda_51;
    double lambda_12, lambda_22, lambda_32, lambda_42, lambda_52;
    double lambda_13, lambda_23, lambda_33, lambda_43, lambda_53;
    double lambda_14, lambda_24, lambda_34, lambda_44, lambda_54;
    double lambda_15, lambda_25, lambda_35, lambda_45, lambda_55;
    double val1, val2, val3, val4, val5;
} LambdaParams ;

EXPORT_SYMBOL void calc_stresses(const PixelData* input,
                                 const LambdaParams* params,
                                 const int rows,
                                 const int cols,
                                 double* stress);

#ifdef __cplusplus
}
#endif
