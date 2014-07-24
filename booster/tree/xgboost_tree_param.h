#ifndef XGBOOST_TREE_PARAM_H
#define XGBOOST_TREE_PARAM_H
/*!
 * \file xgboost_tree_param.h
 * \brief training parameters, statistics used to support tree construction
 * 
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
#include <cstring>

namespace xgboost{
    namespace booster{
        /*! \brief core statistics used for tree construction */
        struct CoreStats{
            /*! \brief sum gradient statistics */
            double sum_grad;
            /*! \brief sum hessian statistics */
            double sum_hess;
            /*! \brief clear the statistics */
            inline void Clear(void){
                sum_grad = sum_hess = 0.0f;
            }
            /*! \brief add statistics to the data */
            inline void Add( double grad, double hess ){
                sum_grad += grad; sum_hess += hess;
            }
            /*! \brief add statistics to the data */
            inline void Add( const CoreStats &b ){
                this->Add(b.sum_grad, b.sum_hess);
            }
            /*! \brief substract the statistics by b */
            inline CoreStats Substract(const CoreStats &b) const{
                CoreStats res;
                res.sum_grad = this->sum_grad - b.sum_grad;
                res.sum_hess = this->sum_hess - b.sum_hess;
                return res;
            }
            inline bool Empty(void) const{
                return sum_hess == 0.0;
            }
        };        
    };

    namespace booster{
        /*! \brief training parameters for regression tree */
        struct TreeParamTrain{
            // learning step size for a time
            float learning_rate;
            // minimum loss change required for a split
            float min_split_loss;
            // maximum depth of a tree
            int   max_depth;
            //----- the rest parameters are less important ----
            // minimum amount of hessian(weight) allowed in a child
            float min_child_weight;
            // weight decay parameter used to control leaf fitting
            float reg_lambda;
            // reg method
            int   reg_method;
            // default direction choice
            int   default_direction;
            // whether we want to do subsample
            float subsample;
            // whether to use layerwise aware regularization
            int   use_layerwise;
            // whether to subsample columns each split, in each level
            float colsample_bylevel;
            // whether to subsample columns during tree construction
            float colsample_bytree;
            // speed optimization for dense column 
            float opt_dense_col;

            // number of threads to be used for tree construction, if OpenMP is enabled, if equals 0, use system default
            int nthread;
            /*! \brief constructor */
            TreeParamTrain( void ){
                learning_rate = 0.3f;
                min_child_weight = 1.0f;
                max_depth = 6;
                reg_lambda = 1.0f;
                reg_method = 2;
                default_direction = 0;
                subsample = 1.0f;
                colsample_bytree = 1.0f;
                colsample_bylevel = 1.0f;
                use_layerwise = 0;
                opt_dense_col = 1.0f;
                nthread = 0;
            }
            /*! 
             * \brief set parameters from outside 
             * \param name name of the parameter
             * \param val  value of the parameter
             */            
            inline void SetParam( const char *name, const char *val ){
                // sync-names 
                if( !strcmp( name, "gamma") )  min_split_loss = (float)atof( val );
                if( !strcmp( name, "eta") )    learning_rate  = (float)atof( val );
                if( !strcmp( name, "lambda") ) reg_lambda = (float)atof( val );
                // normal tree prameters
                if( !strcmp( name, "learning_rate") )     learning_rate = (float)atof( val );
                if( !strcmp( name, "min_child_weight") )  min_child_weight = (float)atof( val );
                if( !strcmp( name, "min_split_loss") )    min_split_loss = (float)atof( val );
                if( !strcmp( name, "max_depth") )         max_depth = atoi( val );           
                if( !strcmp( name, "reg_lambda") )        reg_lambda = (float)atof( val );
                if( !strcmp( name, "reg_method") )        reg_method = (float)atof( val );
                if( !strcmp( name, "subsample") )         subsample  = (float)atof( val );
                if( !strcmp( name, "colsample_bylevel") ) colsample_bylevel  = (float)atof( val );
                if( !strcmp( name, "colsample_bytree") )  colsample_bytree  = (float)atof( val );
                if( !strcmp( name, "use_layerwise") )     use_layerwise = atoi( val );
                if( !strcmp( name, "opt_dense_col") )     opt_dense_col = (float)atof( val );
                if( !strcmp( name, "nthread") )           nthread = atoi( val );
                if( !strcmp( name, "default_direction") ) {
                    if( !strcmp( val, "learn") )  default_direction = 0;
                    if( !strcmp( val, "left") )   default_direction = 1;
                    if( !strcmp( val, "right") )  default_direction = 2;
                }
            }
        protected:
            // functions for L1 cost
            static inline double ThresholdL1( double w, double lambda ){
                if( w > +lambda ) return w - lambda;
                if( w < -lambda ) return w + lambda;
                return 0.0;
            }
            inline double CalcWeight( double sum_grad, double sum_hess )const{
                if( sum_hess < min_child_weight ){
                    return 0.0;
                }else{
                    switch( reg_method ){
                    case 1: return - ThresholdL1( sum_grad, reg_lambda ) / sum_hess;
                    case 2: return - sum_grad / ( sum_hess + reg_lambda );
                        // elstic net
                    case 3: return - ThresholdL1( sum_grad, 0.5 * reg_lambda ) / ( sum_hess + 0.5 * reg_lambda );
                    default: return - sum_grad / sum_hess;
                    }
                }
            }
        private:
            inline static double Sqr( double a ){
                return a * a;
            }
        public:
            // calculate the cost of loss function
            inline double CalcGain( double sum_grad, double sum_hess ) const{
                if( sum_hess < min_child_weight ){
                    return 0.0;
                }
                switch( reg_method ){
                case 1 : return Sqr( ThresholdL1( sum_grad, reg_lambda ) ) / sum_hess;
                case 2 : return Sqr( sum_grad ) / ( sum_hess + reg_lambda );
                    // elstic net
                case 3 : return Sqr( ThresholdL1( sum_grad, 0.5 * reg_lambda ) ) / ( sum_hess + 0.5 * reg_lambda );
                default: return Sqr( sum_grad ) / sum_hess;
                }        
            }
            // KEY:layerwise
            // calculate cost of root
            inline double CalcRootGain( double sum_grad, double sum_hess ) const{
                if( use_layerwise == 0 ) return this->CalcGain( sum_grad, sum_hess );
                else return 0.0;
            }
            // KEY:layerwise
            // calculate the cost after split
            // base_weight: the base_weight of parent           
            inline double CalcGain( double sum_grad, double sum_hess, double base_weight ) const{
                if( use_layerwise == 0 ) return this->CalcGain( sum_grad, sum_hess );
                else return this->CalcGain( sum_grad + sum_hess * base_weight, sum_hess );
            }
            // calculate the weight of leaf
            inline double CalcWeight( double sum_grad, double sum_hess, double parent_base_weight )const{
                if( use_layerwise == 0 ) return CalcWeight( sum_grad, sum_hess );
                else return parent_base_weight + CalcWeight( sum_grad + parent_base_weight * sum_hess, sum_hess );
            }           
            /*! \brief whether need forward small to big search: default right */
            inline bool need_forward_search( float col_density = 0.0f ) const{
                return this->default_direction == 2 || (default_direction == 0 && (col_density<opt_dense_col) );
            }
            /*! \brief whether need backward big to small search: default left */
            inline bool need_backward_search( float col_density = 0.0f ) const{
                return this->default_direction != 2;
            }
            /*! \brief given the loss change, whether we need to invode prunning */
            inline bool need_prune( double loss_chg, int depth ) const{
                return loss_chg < this->min_split_loss;
            }
            /*! \brief whether we can split with current hessian */
            inline bool cannot_split( double sum_hess, int depth ) const{
                return sum_hess < this->min_child_weight * 2.0; 
            }
        public:
            // code support for template data
            inline double CalcWeight( const CoreStats &d, double parent_base_weight )const{
                return this->CalcWeight( d.sum_grad, d.sum_hess, parent_base_weight );                
            }            
            inline double CalcGain( const CoreStats &d, double base_weight ) const{
                return this->CalcGain( d.sum_grad, d.sum_hess, base_weight );
            }
            inline double CalcRootGain( const CoreStats &d ) const{
                return this->CalcRootGain( d.sum_grad, d.sum_hess );
            }
        };
    };
};

#endif
