#ifndef XGBOOST_COL_TREEMAKER_INL_HPP
#define XGBOOST_COL_TREEMAKER_INL_HPP
/*!
 * \file xgboost_col_treemaker-inl.hpp
 * \brief implementation of regression tree maker,
 *        use a column based approach, with OpenMP 
 *        templating the statistics
 * \author Tianqi Chen: tianqi.tchen@gmail.com 
 */
// use openmp
#include <vector>
#include "xgboost_tree_model.h"
#include "../../utils/xgboost_omp.h"
#include "../../utils/xgboost_random.h"
#include "../../utils/xgboost_fmap.h"
#include "xgboost_base_treemaker-inl.hpp"

namespace xgboost{
    namespace booster{
        template<typename FMatrix, typename TStats>
        class ColTreeMakerX : protected BaseTreeMakerX{
        public:
            ColTreeMakerX( RegTree &tree,
                           const TreeParamTrain &param, 
                           const std::vector<float> &grad,
                           const std::vector<float> &hess,
                           const FMatrix &smat, 
                           const std::vector<unsigned> &root_index, 
                           const utils::FeatConstrain  &constrain )
                : BaseTreeMakerX( tree, param ), 
                  grad(grad), hess(hess), 
                  smat(smat), root_index(root_index), constrain(constrain) {
                utils::Assert( grad.size() == hess.size(), "booster:invalid input" );
                utils::Assert( smat.NumRow() == hess.size(), "booster:invalid input" );
                utils::Assert( root_index.size() == 0 || root_index.size() == hess.size(), "booster:invalid input" );                
                utils::Assert( smat.HaveColAccess(), "ColTreeMaker: need column access matrix" );
            }
            inline void Make( int& stat_max_depth, int& stat_num_pruned ){
                this->InitData();
                this->InitNewNode( this->qexpand );
                stat_max_depth = 0;
                
                for( int depth = 0; depth < param.max_depth; ++ depth ){
                    this->FindSplit( depth );
                    this->UpdateQueueExpand( this->qexpand );
                    this->InitNewNode( this->qexpand );
                    // if nothing left to be expand, break
                    if( qexpand.size() == 0 ) break;
                    stat_max_depth = depth + 1;
                }
                // set all the rest expanding nodes to leaf
                for( size_t i = 0; i < qexpand.size(); ++ i ){
                    const int nid = qexpand[i];
                    tree[ nid ].set_leaf( snode[nid].weight * param.learning_rate );                        
                }
                // start prunning the tree
                stat_num_pruned = this->DoPrune();
            }
        private:
            /*! \brief per thread x per node entry to store tmp data */
            struct ThreadEntry{
                /*! \brief statistics of data*/
                TStats stats;
                /*! \brief last feature value scanned */
                float  last_fvalue;
                /*! \brief current best solution */
                SplitEntry best;
                /*! \brief constructor */
                ThreadEntry( void ){                    
                    this->ClearStats();
                }
                /*! \brief clear statistics */
                inline void ClearStats( void ){
                    stats.Clear();
                }
            };
        private:
            // try to prune off current leaf, return true if successful
            inline void TryPruneLeaf( int nid, int depth ){
                if( tree[ nid ].is_root() ) return;
                int pid = tree[ nid ].parent();
                RegTree::NodeStat &s = tree.stat( pid );
                ++ s.leaf_child_cnt;
                
                if( s.leaf_child_cnt >= 2 && param.need_prune( s.loss_chg, depth - 1 ) ){
                    this->stat_num_pruned += 2;
                    // need to be pruned
                    tree.ChangeToLeaf( pid, param.learning_rate * s.base_weight );
                    // tail recursion
                    this->TryPruneLeaf( pid, depth - 1 );
                }
            }
            struct NodeEntry{
                /*! \brief statics for node entry */
                TStats stats;
                /*! \brief loss of this node, without split */
                float  root_gain;
                /*! \brief weight calculated related to current data */
                float  weight;
                /*! \brief current best solution */
                SplitEntry best;
                NodeEntry( void ){
                    stats.Clear();
                    weight = root_gain = 0.0f;
                }
            };
        private:
            // make leaf nodes for all qexpand, update node statistics, mark leaf value
            inline void InitNewNode( const std::vector<int> &qexpand ){
                {// setup statistics space for each tree node
                   for( size_t i = 0; i < stemp.size(); ++ i ){
                        stemp[i].resize( tree.param.num_nodes, ThreadEntry() );
                   }
                    snode.resize( tree.param.num_nodes, NodeEntry() );
                }

                const unsigned ndata = static_cast<unsigned>( position.size() );
                
                #pragma omp parallel for schedule( static )
                for( unsigned i = 0; i < ndata; ++ i ){
                    const int tid = omp_get_thread_num();
                    if( position[i] < 0 ) continue; 
                    stemp[tid][ position[i] ].stats.Add( grad[i], hess[i] );
                }

                for( size_t j = 0; j < qexpand.size(); ++ j ){
                    const int nid = qexpand[ j ];
                    TStats stats; stats.Clear();
                    for( size_t tid = 0; tid < stemp.size(); tid ++ ){
                        stats.Add( stemp[tid][nid].stats );
                    }
                    // update node statistics
                    snode[nid].stats = stats;
                    snode[nid].root_gain = param.CalcRootGain( stats );
                    if( !tree[nid].is_root() ){
                        snode[nid].weight = param.CalcWeight( stats, tree.stat( tree[nid].parent() ).base_weight );
                        tree.stat(nid).base_weight = snode[nid].weight;
                    }else{
                        snode[nid].weight = param.CalcWeight( stats, 0.0f );
                        tree.stat(nid).base_weight = snode[nid].weight;
                    }
                }
            }
        private:
            // enumerate the split values of specific feature
            template<typename Iter>
            inline void EnumerateSplit( Iter it, const unsigned fid, std::vector<ThreadEntry> &temp, bool is_forward_search ){
                // clear all the temp statistics
                for( size_t j = 0; j < qexpand.size(); ++ j ){
                    temp[ qexpand[j] ].ClearStats();
                }
                
                while( it.Next() ){
                    const bst_uint ridx = it.rindex();
                    const int nid = position[ ridx ];
                    if( nid < 0 ) continue;

                    const float fvalue = it.fvalue();           
                    ThreadEntry &e = temp[ nid ];

                    // test if first hit, this is fine, because we set 0 during init
                    if( e.stats.Empty() ){
                        e.stats.Add( grad[ridx], hess[ridx] );
                        e.last_fvalue = fvalue;
                    }else{
                        // try to find a split
                        if( fabsf(fvalue - e.last_fvalue) > rt_2eps && e.stats.sum_hess >= param.min_child_weight ){
                            TStats c = snode[nid].stats.Substract( e.stats );
                            if( c.sum_hess >= param.min_child_weight ){
                                const double loss_chg = 
                                    + param.CalcGain( e.stats, snode[nid].weight ) 
                                    + param.CalcGain( c, snode[nid].weight )
                                    - snode[nid].root_gain;
                                e.best.Update( loss_chg, fid, (fvalue + e.last_fvalue) * 0.5f, !is_forward_search );
                            }
                        }
                        // update the statistics
                        e.stats.Add( grad[ridx], hess[ridx] );
                        e.last_fvalue = fvalue;
                    }
                }
                // finish updating all statistics, check if it is possible to include all sum statistics
                for( size_t i = 0; i < qexpand.size(); ++ i ){
                    const int nid = qexpand[ i ];
                    ThreadEntry &e = temp[ nid ];  
                    TStats c = snode[nid].stats.Substract( e.stats );                  

                    if( e.stats.sum_hess >= param.min_child_weight && c.sum_hess >= param.min_child_weight ){
                        const double loss_chg = 
                            + param.CalcGain( e.stats, snode[nid].weight ) 
                            + param.CalcGain( c, snode[nid].weight )
                            - snode[nid].root_gain;
                        const float delta = is_forward_search ? rt_eps:-rt_eps;
                        e.best.Update( loss_chg, fid, e.last_fvalue + delta, !is_forward_search );
                    }
                }
            }

            // find splits at current level
            inline void FindSplit( int depth ){
                {// created from feat set
                    std::vector<unsigned> feat_set = feat_index;
                    if( param.colsample_bylevel != 1.0f ){
                        random::Shuffle( feat_set );
                        unsigned n = static_cast<unsigned>( param.colsample_bylevel * feat_index.size() );
                        utils::Assert( n > 0, "colsample_bylevel is too small that no feature can be included" );
                        feat_set.resize( n );
                    }
                    const unsigned nsize = static_cast<unsigned>( feat_set.size() );
                    #pragma omp parallel for schedule( dynamic, 1 )
                    for( unsigned i = 0; i < nsize; ++ i ){
                        const unsigned fid = feat_set[i];
                        const int tid = omp_get_thread_num();
                        
                        if( param.need_forward_search( smat.GetColDensity(fid) ) ){
                            this->EnumerateSplit( smat.GetSortedCol(fid), fid, stemp[tid], true );
                        }
                        if( param.need_backward_search( smat.GetColDensity(fid) ) ){
                            this->EnumerateSplit( smat.GetReverseSortedCol(fid), fid, stemp[tid], false );
                        }
                    }
                }
                    
                // after this each thread's stemp will get the best candidates, aggregate results
                for( size_t i = 0; i < qexpand.size(); ++ i ){
                    const int nid = qexpand[ i ];
                    NodeEntry &e = snode[ nid ];
                    for( int tid = 0; tid < this->nthread; ++ tid ){
                        e.best.Update( stemp[ tid ][ nid ].best );
                    }
                    
                    // now we know the solution in snode[ nid ], set split
                    if( e.best.loss_chg > rt_eps ){
                        tree.AddChilds( nid );
                        tree[ nid ].set_split( e.best.split_index(), e.best.split_value, e.best.default_left() );
                    } else{
                        tree[ nid ].set_leaf( e.weight * param.learning_rate );
                    }  
                }

                {// reset position 
                    // step 1, set default direct nodes to default, and leaf nodes to -1, 
                    const unsigned ndata = static_cast<unsigned>( position.size() );
                    #pragma omp parallel for schedule( static )
                    for( unsigned i = 0; i < ndata; ++ i ){
                        const int nid = position[i];
                        if( nid >= 0 ){
                            if( tree[ nid ].is_leaf() ){
                                position[i] = -1;
                            }else{
                                // push to default branch, correct latter
                                position[i] = tree[nid].default_left() ? tree[nid].cleft(): tree[nid].cright();
                            }
                        }
                    }

                    // step 2, classify the non-default data into right places
                    std::vector<unsigned> fsplits;

                    for( size_t i = 0; i < qexpand.size(); ++ i ){
                        const int nid = qexpand[i];
                        if( !tree[nid].is_leaf() ) fsplits.push_back( tree[nid].split_index() );
                    }
                    std::sort( fsplits.begin(), fsplits.end() );
                    fsplits.resize( std::unique( fsplits.begin(), fsplits.end() ) - fsplits.begin() );

                    const unsigned nfeats = static_cast<unsigned>( fsplits.size() );
                    #pragma omp parallel for schedule( dynamic, 1 )
                    for( unsigned i = 0; i < nfeats; ++ i ){
                        const unsigned fid = fsplits[i];
                        for( typename FMatrix::ColIter it = smat.GetSortedCol( fid ); it.Next(); ){
                            const bst_uint ridx = it.rindex();
                            int nid = position[ ridx ];
                            if( nid == -1 ) continue;
                            // go back to parent, correct those who are not default
                            nid = tree[ nid ].parent();
                            if( tree[ nid ].split_index() == fid ){
                                if( it.fvalue() < tree[nid].split_cond() ){
                                    position[ ridx ] = tree[ nid ].cleft();
                                }else{
                                    position[ ridx ] = tree[ nid ].cright();
                                }
                            }
                        }
                    }
                }
            }
        private:
            // initialize temp data structure
            inline void InitData( void ){
                {
                    position.resize( grad.size() );
                    if( root_index.size() == 0 ){
                        std::fill( position.begin(), position.end(), 0 );
                    }else{
                        for( size_t i = 0; i < root_index.size(); ++ i ){
                            position[i] = root_index[i];
                            utils::Assert( root_index[i] < (unsigned)tree.param.num_roots, "root index exceed setting" );
                        }
                    }
                    // mark delete for the deleted datas
                    for( size_t i = 0; i < grad.size(); ++ i ){
                        if( hess[i] < 0.0f ) position[i] = -1;
                    }
                    if( param.subsample < 1.0f - 1e-6f ){
                        for( size_t i = 0; i < grad.size(); ++ i ){
                            if( hess[i] < 0.0f ) continue;
                            if( random::SampleBinary( param.subsample) == 0 ){
                                position[ i ] = -1;
                            }
                        }
                    }
                }
                
                {// initialize feature index
                    unsigned ncol = static_cast<unsigned>( smat.NumCol() );
                    for( unsigned i = 0; i < ncol; ++i ){
                        if( smat.GetSortedCol(i).Next() && constrain.NotBanned(i) ){
                            feat_index.push_back( i );
                        }
                    }
                    unsigned n = static_cast<unsigned>( param.colsample_bytree * feat_index.size() );
                    random::Shuffle( feat_index );
                    utils::Assert( n > 0, "colsample_bytree is too small that no feature can be included" );                    
                    feat_index.resize( n );
                }
                {// setup temp space for each thread
                    if( param.nthread != 0 ){
                        omp_set_num_threads( param.nthread );
                    }
                    #pragma omp parallel
                    {
                        this->nthread = omp_get_num_threads();
                    }

                    // reserve a small space
                    stemp.resize( this->nthread, std::vector<ThreadEntry>() );
                    for( size_t i = 0; i < stemp.size(); ++ i ){
                        stemp[i].reserve( 256 );
                    }
                    snode.reserve( 256 );
                }
                
                {// expand query
                    qexpand.reserve( 256 ); qexpand.clear();
                    for( int i = 0; i < tree.param.num_roots; ++ i ){
                        qexpand.push_back( i );
                    }
                }
            }
        protected:
            /*! \brief do prunning of a tree */
            inline int DoPrune( void ){
                this->stat_num_pruned = 0;
                // initialize auxiliary statistics
                for( int nid = 0; nid < tree.param.num_nodes; ++ nid ){
                    tree.stat( nid ).leaf_child_cnt = 0;
                    tree.stat( nid ).loss_chg = snode[ nid ].best.loss_chg;
                    tree.stat( nid ).sum_hess = static_cast<float>( snode[ nid ].stats.sum_hess );
                }
                for( int nid = 0; nid < tree.param.num_nodes; ++ nid ){
                    if( tree[ nid ].is_leaf() ) this->TryPruneLeaf( nid, tree.GetDepth(nid) );
                }
                return this->stat_num_pruned;
            }
        private:
            // number of omp thread used during training
            int nthread;
            // Per feature: shuffle index of each feature index
            std::vector<unsigned> feat_index;
            // Instance Data: current node position in the tree of each instance
            std::vector<int> position;
            // PerThread x PerTreeNode: statistics for per thread construction
            std::vector< std::vector<ThreadEntry> > stemp;
            /*! \brief TreeNode Data: statistics for each constructed node, the derived class must maintain this */
            std::vector<NodeEntry> snode;
        private:
            const std::vector<float> &grad;
            const std::vector<float> &hess;
            const FMatrix            &smat;
            const std::vector<unsigned> &root_index;
            const utils::FeatConstrain  &constrain;
        };
    };
};
#endif
