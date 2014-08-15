#ifndef XGBOOST_REGRANK_OBJ_HPP
#define XGBOOST_REGRANK_OBJ_HPP
/*!
 * \file xgboost_regrank_obj.hpp
 * \brief implementation of objective functions
 * \author Tianqi Chen, Kailong Chen
 */
#include <vector>
#include <cmath>
#include <limits>
#include <functional>
#include "xgboost_regrank_utils.h"

namespace xgboost{
    namespace regrank{        
        class RegressionObj : public IObjFunction{
        public:
            RegressionObj( int loss_type ){
                loss.loss_type = loss_type;
                scale_pos_weight = 1.0f;
            }
            virtual ~RegressionObj(){}
            virtual void SetParam(const char *name, const char *val){
                if( !strcmp( "loss_type", name ) ) loss.loss_type = atoi( val );
                if( !strcmp( "scale_pos_weight", name ) ) scale_pos_weight = (float)atof( val );
            }
            virtual void GetGradient(const std::vector<float>& preds,  
                                     const DMatrix::Info &info,
                                     int iter,
                                     std::vector<float> &grad, 
                                     std::vector<float> &hess ) {
                utils::Assert( preds.size() == info.labels.size(), "label size predict size not match" );
                grad.resize(preds.size()); hess.resize(preds.size());

                const unsigned ndata = static_cast<unsigned>(preds.size());
                #pragma omp parallel for schedule( static )
                for (unsigned j = 0; j < ndata; ++j){
                    float p = loss.PredTransform(preds[j]);
                    float w = info.GetWeight(j);
                    if( info.labels[j] == 1.0f ) w *= scale_pos_weight;
                    grad[j] = loss.FirstOrderGradient(p, info.labels[j]) * w;
                    hess[j] = loss.SecondOrderGradient(p, info.labels[j]) * w;
                }
            }
            virtual const char* DefaultEvalMetric(void) {
                if( loss.loss_type == LossType::kLogisticClassify ) return "error";
                if( loss.loss_type == LossType::kLogisticRaw ) return "auc";
                return "rmse";
            }
            virtual void PredTransform(std::vector<float> &preds){
                const unsigned ndata = static_cast<unsigned>(preds.size());
                #pragma omp parallel for schedule( static )
                for (unsigned j = 0; j < ndata; ++j){
                    preds[j] = loss.PredTransform( preds[j] );
                }
            }
        protected:
            float scale_pos_weight;
            LossType loss;
        };
    };

    namespace regrank{
        // simple softmax rak
        class SoftmaxRankObj : public IObjFunction{
        public:
            SoftmaxRankObj(void){
                scale_pos_weight = 1.0f;
                walpha = 1.0f;
                num_group = 2;
                num_repeat = 1;
            }
            virtual ~SoftmaxRankObj(){}
            virtual void SetParam(const char *name, const char *val){
                if( !strcmp( "scale_pos_weight", name ) ) scale_pos_weight = (float)atof( val );
                if( !strcmp( "walpha", name ) )  walpha = (float)atof( val );
                if( !strcmp( "softrank:repeat", name) ) num_repeat = (unsigned)atoi( val );
                if( !strcmp( "softrank:group", name) )  num_group = (unsigned)atoi( val );
            }
            virtual void GetGradient(const std::vector<float>& preds,  
                                     const DMatrix::Info &info,
                                     int iter,
                                     std::vector<float> &grad, 
                                     std::vector<float> &hess ) {
                utils::Assert( preds.size() == info.labels.size(), "label size predict size not match" );
                grad.resize(preds.size()); hess.resize(preds.size());
                std::vector<unsigned> tgptr(2, 0); tgptr[1] = preds.size();
                const std::vector<unsigned> &gptr = info.group_ptr.size() == 0 ? tgptr : info.group_ptr;
                utils::Assert( gptr.size() != 0 && gptr.back() == preds.size(), "SoftmaxRank: invalid group file" );
                const unsigned ngroup = static_cast<unsigned>( gptr.size() - 1 );

                for (unsigned k = 0; k < ngroup; ++k){
                    std::vector<unsigned> pos_index, neg_index;
                    for(unsigned j = gptr[k]; j < gptr[k+1]; ++j ){
                        if( info.labels[j] == 1.0f ) {
                            pos_index.push_back( j );
                        }else{
                            
                            neg_index.push_back( j );
                        }
                        grad[j] = hess[j] = 0.0f;
                    }

                    for(unsigned iter = 0; iter < num_repeat; ++ iter){
                        random::Shuffle( pos_index );
                        random::Shuffle( neg_index );
                        const unsigned nneg = neg_index.size();
                        #pragma omp parallel 
                        {
                            std::vector<float> rec( num_group + 1 );
                            #pragma omp for schedule(static)
                            for( unsigned i = 0; i  < nneg - num_group + 1; i += num_group ){
                                for( unsigned j = 0; j < num_group; ++ j ){
                                    rec[j] = preds[ neg_index[ i + j ] ];
                                }
                                const unsigned pindex = pos_index[ (i/num_group)  % pos_index.size() ];
                                rec.back() = preds[ pindex ];
                                const float pw = info.GetWeight( pindex );
                                Softmax( rec );
                                for( unsigned j = 0; j < num_group; ++ j ){
                                    const float p = rec[j];
                                    const unsigned nindex = neg_index[ i + j ];
                                    grad[nindex] += p * pw; 
                                    hess[nindex] += p * (1.0f-p) * pw; 
                                }
                                {// positive statis
                                    const float p = rec.back();
                                    grad[pindex] += (p-1.0f) * pw;
                                    hess[pindex] += p * (1.0f-p) *pw;
                                }
                            }
                        }
                    }
                    const float scale = scale_pos_weight *  static_cast<float>( pos_index.size() * num_group ) / neg_index.size();
                    #pragma omp parallel for schedule(static)                    
                    for(unsigned j = gptr[k]; j < gptr[k+1]; ++j ){
                        grad[j] *= scale; hess[j] *= scale;
                    }
                }
            }
            virtual const char* DefaultEvalMetric(void) {
                return "auc";
            }
        protected:
            float walpha;
            float scale_pos_weight;
            unsigned num_group, num_repeat;
        };
        
        class SoftmaxWeightRankObj: public SoftmaxRankObj{
        public:
            virtual void GetGradient(const std::vector<float>& preds,  
                                     const DMatrix::Info &info,
                                     int iter,
                                     std::vector<float> &grad, 
                                     std::vector<float> &hess ) {
                utils::Assert( preds.size() == info.labels.size(), "label size predict size not match" );
                grad.resize(preds.size()); hess.resize(preds.size());
                std::vector<unsigned> tgptr(2, 0); tgptr[1] = preds.size();
                const std::vector<unsigned> &gptr = info.group_ptr.size() == 0 ? tgptr : info.group_ptr;
                utils::Assert( gptr.size() != 0 && gptr.back() == preds.size(), "SoftmaxRank: invalid group file" );
                const unsigned ngroup = static_cast<unsigned>( gptr.size() - 1 );

                for (unsigned k = 0; k < ngroup; ++k){
                    std::vector<unsigned> pos_index, neg_index;
                    double neg_wsum = 0.0;
                    std::vector<float> neg_wvec;
                    for(unsigned j = gptr[k]; j < gptr[k+1]; ++j ){
                        if( info.labels[j] == 1.0f ) {
                            pos_index.push_back( j );
                        }else{                            
                            neg_index.push_back( j );
                            neg_wsum += info.GetWeight( j );
                            neg_wvec.push_back( neg_wsum );
                        }
                        grad[j] = hess[j] = 0.0f;
                    }
                    
                    #pragma omp parallel 
                    {
                        std::vector<float> rec( this->num_group + 1 );
                        std::vector<unsigned> neg_sample( num_group ); 
                        random::Random rnd; rnd.Seed( iter * 1111 + omp_get_thread_num() );
                        const unsigned nrep = this->num_repeat * neg_index.size();
                        const float scale = this->scale_pos_weight * pos_index.size() / this->num_repeat;

                        random::Shuffle( pos_index );
                        #pragma omp for schedule(static)                      
                        for( unsigned i = 0; i  < nrep; ++ i ){
                            const unsigned pindex = pos_index[ i % pos_index.size() ];                                                               
                            for( unsigned j = 0; j < this->num_group; ++ j ){
                                // sample negative sample
                                float r = rnd.RandDouble() * neg_wsum;
                                size_t idx = std::lower_bound( neg_wvec.begin(), neg_wvec.end(), r ) - neg_wvec.begin();
                                if( idx == neg_wvec.size() ) idx = neg_wvec.size() - 1;
                                neg_sample[j] = neg_index[idx];
                                // finish generate neg sample
                                rec[j] = preds[ neg_sample[j] ];
                            }                                
                            rec.back() = preds[ pindex ];
                            const float pw = info.GetWeight( pindex ) * scale;                                
                            Softmax( rec );
                            for( unsigned j = 0; j < this->num_group; ++ j ){
                                const float p = rec[j];
                                const unsigned nindex = neg_sample[j];
                                grad[nindex] += p * pw; 
                                hess[nindex] += p * (1.0f-p) * pw; 
                            }
                            {// positive statis
                                const float p = rec.back();
                                grad[pindex] += (p-1.0f) * pw;
                                hess[pindex] += p * (1.0f-p) *pw;
                            }
                        }
                    }
                }
            }
        };
        
        // simple softmax multi-class classification
        class SoftmaxMultiClassObj : public IObjFunction{
        public:
            SoftmaxMultiClassObj(int output_prob):output_prob(output_prob){
                nclass = 0;
            }
            virtual ~SoftmaxMultiClassObj(){}
            virtual void SetParam(const char *name, const char *val){
                if( !strcmp( "num_class", name ) ) nclass = atoi(val); 
            }
            virtual void GetGradient(const std::vector<float>& preds,  
                                     const DMatrix::Info &info,
                                     int iter,
                                     std::vector<float> &grad, 
                                     std::vector<float> &hess ) {
                utils::Assert( nclass != 0, "must set num_class to use softmax" );
                utils::Assert( preds.size() == (size_t)nclass * info.labels.size(), "SoftmaxMultiClassObj: label size and pred size does not match" );
                grad.resize(preds.size()); hess.resize(preds.size());
                
                const unsigned ndata = static_cast<unsigned>(info.labels.size());
                #pragma omp parallel
                {
                    std::vector<float> rec(nclass);
                    #pragma omp for schedule(static)
                    for (unsigned j = 0; j < ndata; ++j){
                        for( int k = 0; k < nclass; ++ k ){
                            rec[k] = preds[j + k * ndata];
                        }
                        Softmax( rec );
                        int label = static_cast<int>(info.labels[j]);
                        if( label < 0 ){
                            label = -label - 1;
                        }
                        utils::Assert( label < nclass, "SoftmaxMultiClassObj: label exceed num_class" );
                        for( int k = 0; k < nclass; ++ k ){
                            float p = rec[ k ];
                            if( label == k ){
                                grad[j+k*ndata] = p - 1.0f;
                            }else{
                                grad[j+k*ndata] = p;
                            }
                            hess[j+k*ndata] = 2.0f * p * ( 1.0f - p );
                        }  
                    }
                }
            }
            virtual void PredTransform(std::vector<float> &preds){
                this->Transform(preds, output_prob);
            }
            virtual void EvalTransform(std::vector<float> &preds){
                this->Transform(preds, 0);
            }
        private:
            inline void Transform(std::vector<float> &preds, int prob){
                utils::Assert( nclass != 0, "must set num_class to use softmax" );
                utils::Assert( preds.size() % nclass == 0, "SoftmaxMultiClassObj: label size and pred size does not match" );                
                const unsigned ndata = static_cast<unsigned>(preds.size()/nclass);
                
                #pragma omp parallel
                {
                    std::vector<float> rec(nclass);
                    #pragma omp for schedule(static)
                    for (unsigned j = 0; j < ndata; ++j){
                        for( int k = 0; k < nclass; ++ k ){
                            rec[k] = preds[j + k * ndata];
                        }
                        if( prob == 0 ){
                            preds[j] = FindMaxIndex( rec );
                        }else{
                            Softmax( rec );
                            for( int k = 0; k < nclass; ++ k ){
                                preds[j + k * ndata] = rec[k];
                            }
                        }
                    }
                }
                if( prob == 0 ){
                    preds.resize( ndata );
                }
            }
            virtual const char* DefaultEvalMetric(void) {
                return "merror";
            }
        private:
            int nclass;
            int output_prob;
        };
    };


    namespace regrank{
        /*! \brief objective for lambda rank */
        class LambdaRankObj : public IObjFunction{
        public:
            LambdaRankObj(void){
                loss.loss_type = LossType::kLogisticRaw;
                fix_list_weight = 0.0f;
                num_pairsample = 1;
            }
            virtual ~LambdaRankObj(){}
            virtual void SetParam(const char *name, const char *val){
                if( !strcmp( "loss_type", name ) )       loss.loss_type = atoi( val );
                if( !strcmp( "fix_list_weight", name ) ) fix_list_weight = (float)atof( val );
                if( !strcmp( "num_pairsample", name ) )  num_pairsample = atoi( val );
            }
        public:
            virtual void GetGradient(const std::vector<float>& preds,  
                                     const DMatrix::Info &info,
                                     int iter,
                                     std::vector<float> &grad, 
                                     std::vector<float> &hess ) {
                utils::Assert( preds.size() == info.labels.size(), "label size predict size not match" );              
                grad.resize(preds.size()); hess.resize(preds.size());
                const std::vector<unsigned> &gptr = info.group_ptr;
                utils::Assert( gptr.size() != 0 && gptr.back() == preds.size(), "rank loss must have group file" );
                const unsigned ngroup = static_cast<unsigned>( gptr.size() - 1 );

                #pragma omp parallel
                {
                    // parall construct, declare random number generator here, so that each 
                    // thread use its own random number generator, seed by thread id and current iteration
                    random::Random rnd; rnd.Seed( iter * 1111 + omp_get_thread_num() );
                    std::vector<LambdaPair> pairs;
                    std::vector<ListEntry>  lst;
                    std::vector< std::pair<float,unsigned> > rec;
                    
                    #pragma omp for schedule(static)
                    for (unsigned k = 0; k < ngroup; ++k){
                        lst.clear(); pairs.clear(); 
                        for(unsigned j = gptr[k]; j < gptr[k+1]; ++j ){
                            lst.push_back( ListEntry(preds[j], info.labels[j], j ) );
                            grad[j] = hess[j] = 0.0f;
                        }                        
                        std::sort( lst.begin(), lst.end(), ListEntry::CmpPred );
                        rec.resize( lst.size() );
                        for( unsigned i = 0; i < lst.size(); ++i ){
                            rec[i] = std::make_pair( lst[i].label, i );
                        }
                        std::sort( rec.begin(), rec.end(), CmpFirst );
                        // enumerate buckets with same label, for each item in the lst, grab another sample randomly
                        for( unsigned i = 0; i < rec.size(); ){
                            unsigned j = i + 1;
                            while( j < rec.size() && rec[j].first == rec[i].first ) ++ j;
                            // bucket in [i,j), get a sample outside bucket
                            unsigned nleft = i, nright = rec.size() - j;
                            if( nleft + nright != 0 ){
                                int nsample = num_pairsample;
                                while( nsample -- ){
                                    for( unsigned pid = i; pid < j; ++ pid ){
                                        unsigned ridx = static_cast<unsigned>( rnd.RandDouble() * (nleft+nright) );
                                        if( ridx < nleft ){
                                            pairs.push_back( LambdaPair( rec[ridx].second, rec[pid].second ) );
                                        }else{
                                            pairs.push_back( LambdaPair( rec[pid].second, rec[ridx+j-i].second ) );
                                        }
                                    }      
                                }
                            }
                            i = j;
                        }
                        // get lambda weight for the pairs
                        this->GetLambdaWeight( lst, pairs );
                        // rescale each gradient and hessian so that the lst have constant weighted
                        float scale = 1.0f / num_pairsample;
                        if( fix_list_weight != 0.0f ){
                            scale *= fix_list_weight / (gptr[k+1] - gptr[k]);
                        }
                        for( size_t i = 0; i < pairs.size(); ++ i ){
                            const ListEntry &pos = lst[ pairs[i].pos_index ];
                            const ListEntry &neg = lst[ pairs[i].neg_index ];
                            const float w = pairs[i].weight * scale;
                            float p = loss.PredTransform( pos.pred - neg.pred );
                            float g = loss.FirstOrderGradient( p, 1.0f );
                            float h = loss.SecondOrderGradient( p, 1.0f );
                            // accumulate gradient and hessian in both pid, and nid, 
                            grad[ pos.rindex ] += g * w; 
                            grad[ neg.rindex ] -= g * w;
                            // take conservative update, scale hessian by 2
                            hess[ pos.rindex ] += 2.0f * h * w; 
                            hess[ neg.rindex ] += 2.0f * h * w;
                        }                       
                    }
                }
            }
            virtual const char* DefaultEvalMetric(void) {
                return "map";
            }
        private:
            // loss function
            LossType loss;
            // number of samples peformed for each instance
            int num_pairsample;            
            // fix weight of each elements in list
            float fix_list_weight;
        protected:
            /*! \brief helper information in a list */
            struct ListEntry{
                /*! \brief the predict score we in the data */
                float pred;
                /*! \brief the actual label of the entry */
                float label;
                /*! \brief row index in the data matrix */                
                unsigned rindex;
                // constructor
                ListEntry(float pred, float label, unsigned rindex): pred(pred),label(label),rindex(rindex){}
                // comparator by prediction
                inline static bool CmpPred(const ListEntry &a, const ListEntry &b){
                    return a.pred > b.pred;
                }
                // comparator by label
                inline static bool CmpLabel(const ListEntry &a, const ListEntry &b){
                    return a.label > b.label;
                }
            };
            /*! \brief a pair in the lambda rank */
            struct LambdaPair{
                /*! \brief positive index: this is a position in the list */
                unsigned pos_index;
                /*! \brief negative index: this is a position in the list */
                unsigned neg_index;
                /*! \brief weight to be filled in */
                float weight;
                LambdaPair( unsigned pos_index, unsigned neg_index ):pos_index(pos_index),neg_index(neg_index),weight(1.0f){}
            };            
            /*! 
             * \brief get lambda weight for existing pairs 
             * \param list a list that is sorted by pred score
             * \param pairs record of pairs, containing the pairs to fill in weights
             */
            virtual void GetLambdaWeight( const std::vector<ListEntry> &sorted_list, std::vector<LambdaPair> &pairs ) = 0;
        };
    };
    
    namespace regrank{
        class PairwiseRankObj: public LambdaRankObj{
        public:
            virtual ~PairwiseRankObj(void){}
            virtual void GetLambdaWeight( const std::vector<ListEntry> &sorted_list, std::vector<LambdaPair> &pairs ){}            
        };

        class LambdaRankObj_NDCG : public LambdaRankObj{            
        public:
            virtual ~LambdaRankObj_NDCG(void){}
            virtual void GetLambdaWeight(const std::vector<ListEntry> &sorted_list, std::vector<LambdaPair> &pairs){
                float IDCG;
                {
                    std::vector<float> labels(sorted_list.size());
                    for (size_t i = 0; i < sorted_list.size(); i++){
                        labels[i] = sorted_list[i].label;
                    }
                    std::sort(labels.begin(), labels.end(), std::greater<float>());
                    IDCG = CalcDCG(labels);
                }

                if( IDCG == 0.0 ){
                    for (size_t i = 0; i < pairs.size(); ++i){
                        pairs[i].weight = 0.0f;
                    }
                }else{
                    IDCG = 1.0f / IDCG;
                    for (size_t i = 0; i < pairs.size(); ++i){                    
                        unsigned pos_idx = pairs[i].pos_index;
                        unsigned neg_idx = pairs[i].neg_index;
                        float pos_loginv = 1.0f / logf(pos_idx+2.0f);
                        float neg_loginv = 1.0f / logf(neg_idx+2.0f);
                        int pos_label = static_cast<int>(sorted_list[pos_idx].label);
                        int neg_label = static_cast<int>(sorted_list[neg_idx].label);
                        float original = ((1<<pos_label)-1) * pos_loginv + ((1<<neg_label)-1) * neg_loginv;
                        float changed  = ((1<<neg_label)-1) * pos_loginv + ((1<<pos_label)-1) * neg_loginv;
                        float delta = (original-changed) * IDCG;
                        if( delta < 0.0f ) delta = - delta;
                        pairs[i].weight = delta;
                    }
                }
            }
        private:
            inline static float CalcDCG( const std::vector<float> &labels ){
                double sumdcg = 0.0;
                for( size_t i = 0; i < labels.size(); i ++ ){
                    const unsigned rel = labels[i];
                    if( rel != 0 ){ 
                        sumdcg += ((1<<rel)-1) / logf( i + 2 );
                    }
                }
                return static_cast<float>(sumdcg);
            }
        };

        class LambdaRankObj_MAP : public LambdaRankObj{

            struct MAPStats{
            
                /* \brief the accumulated precision */
                float ap_acc;
                /* \brief the accumulated precision assuming a positive instance is missing*/
                float ap_acc_miss;
                /* \brief the accumulated precision assuming that one more positive instance is inserted ahead*/
                float ap_acc_add;
                /* \brief the accumulated positive instance count */
                float hits;
                
                MAPStats(){}
                
                MAPStats(float ap_acc, float ap_acc_miss, float ap_acc_add, float hits
                    ) :ap_acc(ap_acc), ap_acc_miss(ap_acc_miss), ap_acc_add(ap_acc_add), hits(hits){

                }

            };

        public:
            virtual ~LambdaRankObj_MAP(void){}

            /*
            * \brief Obtain the delta MAP if trying to switch the positions of instances in index1 or index2
            *        in sorted triples
            * \param sorted_list the list containing entry information
            * \param index1,index2 the instances switched
            * \param map_stats a vector containing the accumulated precisions for each position in a list
            */
            inline float GetLambdaMAP(const std::vector<ListEntry> &sorted_list,
                int index1, int index2,
                std::vector< MAPStats > &map_stats){
                if (index1 == index2 || map_stats[map_stats.size() - 1].hits == 0) {
                    return 0.0;
                }
                if (index1 > index2) std::swap(index1, index2);
                float original = map_stats[index2].ap_acc;
                if (index1 != 0) original -= map_stats[index1 - 1].ap_acc;
                float changed = 0, label1 = sorted_list[index1].label > 0?1:0,label2 = sorted_list[index2].label > 0?1:0;
                if(label1 == label2){
                    return 0.0;
                }else if (label1 < label2){
                    changed += map_stats[index2 - 1].ap_acc_add - map_stats[index1].ap_acc_add;
                    changed += (map_stats[index1].hits + 1.0f) / (index1 + 1);
                }
                else{
                    changed += map_stats[index2 - 1].ap_acc_miss - map_stats[index1].ap_acc_miss;
                    changed += map_stats[index2].hits / (index2 + 1);
                }

                float ans = (changed - original) / (map_stats[map_stats.size() - 1].hits);
                if (ans < 0) ans = -ans;
                return ans;
            }

            /*
            * \brief obtain preprocessing results for calculating delta MAP
            * \param sorted_list the list containing entry information
            * \param map_stats a vector containing the accumulated precisions for each position in a list
            */
            inline void GetMAPStats(const std::vector<ListEntry> &sorted_list,
                std::vector< MAPStats > &map_acc){
                map_acc.resize(sorted_list.size());
                float hit = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                for (size_t i = 1; i <= sorted_list.size(); i++){
                    if ((int)sorted_list[i - 1].label > 0) {
                        hit++;
                        acc1 += hit / i;
                        acc2 += (hit - 1) / i;
                        acc3 += (hit + 1) / i;
                    }

                    map_acc[i - 1] = MAPStats(acc1,acc2,acc3,hit);
                }
            }

            virtual void GetLambdaWeight(const std::vector<ListEntry> &sorted_list, std::vector<LambdaPair> &pairs){
                std::vector< MAPStats > map_stats;
                GetMAPStats(sorted_list, map_stats);
                for (size_t i = 0; i < pairs.size(); i++){
                    pairs[i].weight = GetLambdaMAP(sorted_list, pairs[i].pos_index, pairs[i].neg_index, map_stats);
                }
            }
           
        };

    };
};
#endif
