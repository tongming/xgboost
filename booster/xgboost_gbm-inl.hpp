#ifndef XGBOOST_GBM_INL_HPP
#define XGBOOST_GBM_INL_HPP
#include "xgboost_gbmbase.h"

/*!
 * \file xgboost_gbm-inl.hpp
 * \brief instable GBMBase to handle some complicated cases, not needed usually
 *        fall back to GBMBase, used to test extendibilty of GBM
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
namespace xgboost{
    namespace booster{
        class GBMPP: public GBMBase{
        public:
            GBMPP(void){}
            virtual ~GBMPP(void){
            }
            virtual void SetParam(const char *name, const char *val){
                GBMBase::SetParam(name, val);
            }
            virtual void LoadModel(utils::IStream &fi){
                GBMBase::LoadModel(fi);
                if( mparam.num_extend_group == 0 ) return;
                mexts.resize( mparam.num_boosters );
                for(size_t i = 0; i < mexts.size(); ++i){
                    mexts[i].LoadModel(fi);
                }
                if(mparam.num_pbuffer != 0){
                    pred_root.resize(mparam.PredBufferSize());
                    utils::Assert(fi.Read(&pred_root[0], pred_root.size()*sizeof(unsigned)) != 0);
                }
            }            
            virtual void SaveModel(utils::IStream &fo) const {
                GBMBase::SaveModel(fo);
                if( mparam.num_extend_group == 0 ) return;
                for(size_t i = 0; i < mexts.size(); ++i){
                    mexts[i].SaveModel(fo);
                }
                if(mparam.num_pbuffer != 0){
                    fo.Write(&pred_root[0], pred_root.size()*sizeof(unsigned));
                }
            }
            virtual void InitModel(void){
                GBMBase::InitModel();
                if( mparam.num_extend_group == 0 ) return;                
                pred_root.resize(mparam.PredBufferSize(), 0);
            }            
        public:
            virtual void DoBoost(std::vector<float> &grad,
                                 std::vector<float> &hess,
                                 const booster::FMatrixS &feats,
                                 const std::vector<unsigned> &root_index,
                                 int bst_group = 0, int buffer_offset = -1 ) {
                if( mparam.num_extend_group == 0 ){
                    GBMBase::DoBoost(grad, hess, feats, root_index, bst_group, buffer_offset ); return;
                }
                //------------------------------------
                utils::Assert( buffer_offset >= 0, "buffer offset" );
                {// add booster
                    mparam.num_boosters += 1;
                    boosters.push_back(booster::CreateBooster<FMatrixS>(mparam.booster_type));
                    booster_info.push_back(bst_group);
                    this->ConfigBooster(boosters.back());
                    if( boosters.size() != 1 ){
                        char s[16]; sprintf(s, "%d", mparam.num_extend_group );
                        boosters.back()->SetParam("num_roots", s );
                    }
                    boosters.back()->InitModel();
                }
                const unsigned ndata = static_cast<unsigned>(grad.size());
                std::vector<unsigned> root_set(ndata);                  
                #pragma omp parallel for schedule( static )
                for(unsigned i = 0; i < ndata; ++i ){
                    const int bid = mparam.BufferOffset(buffer_offset+i, bst_group);
                    utils::Assert( this->pred_counter[bid]+1 == (unsigned)mparam.num_boosters, "BUGC" );
                    root_set[i] = this->pred_root[bid];
                }
                boosters.back()->DoBoost(grad, hess, feats, root_set);                
                {// add extmodel
                    mexts.push_back( ExtModel() ); 
                    #pragma omp parallel 
                    {
                        ExtModel mdl;
                        #pragma omp for schedule( static )
                        for(unsigned i = 0; i < ndata; ++i ){
                            std::vector<int> path;
                            boosters.back()->PredPath(path, feats, i, root_set[i] );
                            unsigned tid = path.back();
                            if( tid >= mdl.root_map.size() ){
                                mdl.root_map.resize(tid + 1, -1);
                                mdl.leaf_value.resize(tid + 1, 0.0f);
                            }
                            mdl.root_map[tid] = 0; 
                            mdl.leaf_value[tid] = boosters.back()->Predict(feats, i, root_set[i]);
                        }
                        #pragma omp critical
                        {
                            ExtModel &global = mexts.back();
                            if( global.root_map.size() < mdl.root_map.size() ){
                                global.root_map.resize(mdl.root_map.size(), -1);
                                global.leaf_value.resize(mdl.leaf_value.size());
                            }
                            for( size_t i = 0; i < mdl.root_map.size(); ++ i ) {
                                if( mdl.root_map[i] == 0 ){
                                    global.root_map[i] = 0; global.leaf_value[i] = mdl.leaf_value[i];
                                }
                            }
                        }
                    }
                    ExtModel &global = mexts.back();
                    for( size_t i = 0; i < global.root_map.size(); ++i ){
                        if( global.root_map[i] == 0 ){
                            global.root_map[i] = random::NextUInt32(mparam.num_extend_group);
                            utils::Assert( global.root_map[i] < mparam.num_extend_group, "BUGA");
                        }
                    }
                }
            }
            virtual float Predict(const FMatrixS &feats, bst_uint row_index, 
                                  int buffer_index = -1, unsigned root_index = 0, int bst_group = 0 ){
                if( mparam.num_extend_group == 0 ){
                    return GBMBase::Predict(feats, row_index, buffer_index, root_index, bst_group );
                }
                utils::Assert( root_index == 0, "extension restriction");
                size_t itop = 0;
                float  psum = 0.0f;
                const int bid = mparam.BufferOffset(buffer_index, bst_group);

                // load buffered results if any
                if (mparam.do_reboost == 0 && bid >= 0){
                    itop = this->pred_counter[bid];
                    psum = this->pred_buffer[bid];
                    root_index = pred_root[bid];
                }

                for(size_t i = itop; i < this->boosters.size(); ++i ){                    
                    if( booster_info[i] == bst_group ){
                        psum += this->boosters[i]->Predict(feats, row_index, root_index);
                        std::vector<int> path;
                        this->boosters[i]->PredPath(path, feats, row_index, root_index );                        
                        int ridx = mexts[i].root_map[ path.back() ];
                        utils::Assert( ridx >= 0, "BUG");
                        root_index = static_cast<unsigned>(ridx);
                    }
                }
                // updated the buffered results
                if (mparam.do_reboost == 0 && bid >= 0){
                    this->pred_counter[bid] = static_cast<unsigned>(boosters.size());
                    this->pred_buffer[bid] = psum;
                    this->pred_root[bid] = root_index;
                }
                return psum;
            }
        protected:
            // extended model
            struct ExtModel{                
                std::vector<int>   root_map;
                std::vector<float> leaf_value;                
                inline void LoadModel(utils::IStream &fi){
                    unsigned sz;
                    utils::Assert(fi.Read(&sz, sizeof(unsigned)) != 0, "ExtModel" );
                    root_map.resize(sz); leaf_value.resize(sz);
                    utils::Assert(fi.Read(&root_map[0], sizeof(int)*root_map.size()) != 0, "ExtModel" );
                    utils::Assert(fi.Read(&leaf_value[0], sizeof(float)*root_map.size()) != 0, "ExtModel" );                    
                }
                inline void SaveModel(utils::IStream &fo)const{
                    unsigned sz = root_map.size();
                    utils::Assert( sz == leaf_value.size() );
                    fo.Write(&sz, sizeof(unsigned));
                    fo.Write(&root_map[0], sizeof(int)*root_map.size());
                    fo.Write(&leaf_value[0], sizeof(float)*root_map.size());
                }
            };
        protected:
            std::vector<ExtModel> mexts;
            /*! \brief prediction buffer */
            std::vector<unsigned> pred_root;
        };
    };
};

#endif
