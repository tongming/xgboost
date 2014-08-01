#ifndef XGBOOST_BASE_TREEMAKER_INL_HPP
#define XGBOOST_BASE_TREEMAKER_INL_HPP
/*!
 * \file xgboost_base_treemaker-inl.hpp
 * \brief implementation of base data structure for regression tree maker,
 *         gives common operations of tree construction steps template 
 *         an updated version templating the statistics you need
 * 
 * \author Tianqi Chen: tianqi.tchen@gmail.com 
 */
#include <vector>
#include "xgboost_tree_model.h"

namespace xgboost{
    namespace booster{
        // templating BaseTreeMaker
        class BaseTreeMakerX{
        protected:
            BaseTreeMakerX( RegTree &tree,
                            const TreeParamTrain &param )
                : tree( tree ), param( param ){}
        protected:
            // statistics that is helpful to decide a split
            struct SplitEntry{
                /*! \brief loss change after split this node */
                float  loss_chg;
                /*! \brief split index */
                unsigned  sindex;
                /*! \brief split value */
                float     split_value;
                /*! \brief constructor */
                SplitEntry( void ){
                    loss_chg = 0.0f;
                    split_value = 0.0f; sindex = 0;
                }
                // This function gives better priority to lower index when loss_chg equals
                // not the best way, but helps to give consistent result during multi-thread execution
                inline bool NeedReplace( float loss_chg, unsigned split_index ) const{
                    if( this->split_index() <= split_index ){
                        return loss_chg > this->loss_chg; 
                    }else{
                        return !(this->loss_chg > loss_chg);
                    }
                }
                inline bool Update( const SplitEntry &e ){
                    if( this->NeedReplace( e.loss_chg, e.split_index() ) ){
                        this->loss_chg = e.loss_chg;
                        this->sindex = e.sindex;
                        this->split_value = e.split_value;
                        return true;
                    } else{
                        return false;
                    }
                }
                inline bool Update( float loss_chg, unsigned split_index, float split_value, bool default_left ){                    
                    if( this->NeedReplace( loss_chg, split_index ) ){
                        this->loss_chg = loss_chg;
                        if( default_left ) split_index |= (1U << 31);
                        this->sindex = split_index;
                        this->split_value = split_value;
                        return true;
                    }else{
                        return false;
                    }
                }
                inline unsigned split_index( void ) const{
                    return sindex & ( (1U<<31) - 1U );
                }
                inline bool default_left( void ) const{
                    return (sindex >> 31) != 0;
                }
            };
        protected:
            /*! \brief update queue expand add in new leaves */
            inline void UpdateQueueExpand( std::vector<int> &qexpand ){
                std::vector<int> newnodes;
                for( size_t i = 0; i < qexpand.size(); ++ i ){
                    const int nid = qexpand[i];
                    if( !tree[ nid ].is_leaf() ){
                        newnodes.push_back( tree[nid].cleft() );
                        newnodes.push_back( tree[nid].cright() );
                    }
                }
                // use new nodes for qexpand
                qexpand = newnodes;
            }
        protected:
            // local helper tmp data structure
            // statistics
            int stat_num_pruned;
            /*! \brief queue of nodes to be expanded */
            std::vector<int> qexpand;
        protected:
            // original data that supports tree construction
            RegTree &tree;
            const TreeParamTrain &param;
        };
    }; // namespace booster
}; // namespace xgboost
#endif // XGBOOST_BASE_TREEMAKER_INL_HPP
