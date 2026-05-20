#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <limits>
#include <sstream>
#include <regex>
#include <limits>
#include <type_traits>
#include <array>

#include "aris/dynamic/model.hpp"
#include "aris/core/reflection.hpp"

namespace aris::dynamic{
	struct Solver::Imp{
		int which_root_{ 0 }, root_num_{ 1 };
		double max_error_{ 1e-10 };
		Size max_iter_count_{ 100 };
		double error_{ 0.0 };
		Size iter_count_{ 0 };
		Imp(Size max_iter_count, double max_error) :max_iter_count_(max_iter_count), max_error_(max_error) {};
	};
	auto Solver::setRootNumber(int root_num)->void { 
		imp_->root_num_ = root_num;
	}
	auto Solver::rootNumber()const->int { return imp_->root_num_; }
	auto Solver::setWhichRoot(int root_of_solver)->void { imp_->which_root_ = root_of_solver; }
	auto Solver::whichRoot()const->int { return imp_->which_root_; }
	
	auto Solver::error()const->double { return imp_->error_; }
	auto Solver::setError(double error)->void { imp_->error_ = error; }
	auto Solver::maxError()const->double { return imp_->max_error_; }
	auto Solver::setMaxError(double max_error)->void { imp_->max_error_ = max_error; }
	auto Solver::iterCount()const->Size { return imp_->iter_count_; }
	auto Solver::setIterCount(Size iter_count)->void { imp_->iter_count_ = iter_count; }
	auto Solver::maxIterCount()const->Size { return imp_->max_iter_count_; }
	auto Solver::setMaxIterCount(Size max_count)->void { imp_->max_iter_count_ = max_count; }
	Solver::~Solver() = default;
	Solver::Solver(Size max_iter_count, double max_error) : imp_(new Imp(max_iter_count, max_error)) {}
	ARIS_DEFINE_BIG_FOUR_CPP(Solver);

#define ARIS_LOOP_BLOCK(RELATION) for (auto b = RELATION blk_data_; b < RELATION blk_data_ + RELATION blk_size_; ++b)
#define ARIS_LOOP_D for (auto d = d_data_; d < d_data_ + d_size_; ++d)
#define ARIS_LOOP_D_2_TO_END for (auto d = d_data_ + 1; d < d_data_ + d_size_; ++d)
#define ARIS_LOOP_DIAG_INVERSE_2_TO_END for (auto d = d_data_ + d_size_ - 1; d > d_data_; --d)
#define ARIS_LOOP_R for (auto r = r_data_; r < r_data_ + r_size_; ++r)

	struct Relation{
		struct Block { 
			const Constraint* cst_;
			bool is_I_;
			int mot_dim_pos_; // Record the dim pos of motion, if -1, it's not a motion
			int mot_mp_pos_;  // Record the idx of the mp of motion (since pSize is different from dim)
			double* mp_;
		};

		// Pointers to the two parts of the relation
		const Part *prtI_, *prtJ_; // prtI is the part for the diagonal block
		
		// dim_ variable represents the max dim among the constraints of the subsystem.
		//      It's sorted during initialization, putting the maximum one first.
		// size_ variable represents the sum of dim() of all constraints on both sides.
		Size dim_, size_;
		
		Block* blk_data_;
		Size blk_size_;
	};
	struct LocalRelation :public Relation { std::vector<Block> cst_pool_; }; // Just for implementation
	// For Diag and Remainder, see Step4 in the Imp comments, which divides the solved yp and transformed C into two parts
	struct Diag{
		// D * C * P =[I  C]
		//            [0  0]
		// P makes sense for relations with multiple constraints
		Size *p_;
		double dm_[36], iv_[10];
		double pm1_[16], pm2_[16], *pm_, *last_pm_;
		double xp_[6], bp_[6], *bc_, *xc_;
		double *cmI_, *cmJ_, *cmU_, *cmT_; // Since there could be multiple constraints, the total number of constraints might exceed 6, and the dimension of cm is unknown

		Size rows_; // number of row in F
		const Part *part_;
		Diag *rd_; // related diag, for row addition
		Relation rel_;

		typedef void(*UpdFunc2)(Diag*, bool cpt_cp);
		UpdFunc2 upd_d_and_cp_;
	};
	// For Diag and Remainder, see Step4 in the Imp comments, which divides the solved yp and transformed C into two parts
	struct Remainder{
		struct Block { Diag* diag_; bool is_I_; };
		Diag *i_diag_, *j_diag_;
		double *cmI_, *cmJ_, *bc_, *xc_;

		Block* blk_data_;
		Size blk_size_;

		Relation rel_;
	};
	struct LocalRemainder : public Remainder { std::vector<Block> cm_blk_series; };
	struct PublicData;
	struct SubSystem{
		PublicData *pd_;
		
		Diag* d_data_;
		Size d_size_;
		
		Remainder* r_data_;
		Size r_size_;

		Size fm_, fn_, fr_, gm_, gn_;

		bool has_ground_;
		double error_, max_error_;
		Size iter_count_, max_iter_count_;

		auto hasGround()const noexcept->bool { return has_ground_; }
		// Update data from the model //
		auto updDmCm(bool cpt_cp)noexcept->void;
		auto updDiagIv()noexcept->void;
		// Update Cv data, related to kinVel()
		auto updCv()noexcept->void;
		// Update Ca, which is bc
		auto updCa()noexcept->void;
		// Solving section //
		auto updF()noexcept->void;
		// Solve C' * xp = bc, which is A x = b
		auto sovXp()noexcept->void;
		// Update the G matrix described in Step6
		auto updG()noexcept->void;
		// Should only be used by dynamics solver
		auto sovXc()noexcept->void;
		auto sovProjectMassMatrix()noexcept->void;
		auto sovXcRemain()noexcept->void;
		// Interface //
		auto kinPos()noexcept->void;
		auto kinVel()noexcept->void;
		auto dynAccAndFce()noexcept->void;
		auto cptJacobiBeta(std::vector<Size>& targetBodies, 
		std::vector<double>& target_xp)noexcept->void;
	};
	struct PublicData{
		// Active motions //
		aris::dynamic::MotionBase** active_mots_;
		double* active_mp_;
		int active_mot_size_,
			active_mp_size_,
			active_mot_dim_;

		// Inactive motions //
		aris::dynamic::MotionBase** deactive_mots_;
		double* deactive_mp_;
		int deactive_mot_size_,
			deactive_mp_size_,
			deactive_mot_dim_;

		// sys data //
		double gravity_[6];
		
		SubSystem *subsys_data_;
		Size subsys_size_;
		Diag** get_diag_from_part_id_;

		// Jacobian matrix // 
		double* Jg_, * cg_;
		Size mJg_, nJg_;

		// Dynamic matrix // 
		double* M_, * h_;
		Size nM_;

		// Computing memory //
		double *F_, *FU_, *FT_, *G_, *GU_, *GT_, *S_, *QT_DOT_G_, *xpf_, *xcf_, *bpf_, *bcf_, *beta_, *cmI_, *cmJ_, *cmU_, *cmT_;
		Size *FP_, *GP_;
		
	};
	auto SubSystem::updDmCm(bool cpt_cp)noexcept->void{
		if (cpt_cp)error_ = 0.0;// error //
		
		// upd dm and rel dim
		fm_ = 0;
		ARIS_LOOP_D_2_TO_END{
			d->upd_d_and_cp_(d, cpt_cp);// cp //
			d->rows_ = fm_;
			fm_ += 6 - d->rel_.dim_;
			// Get the max error from the errors stored in bc_.
			if (cpt_cp)for (Size i{ 0 }; i < d->rel_.size_; ++i) error_ = std::max(error_, std::abs(d->bc_[i]));// error //
		}

		// upd remainder data //
		ARIS_LOOP_R{
			Size pos{ 0 };
			ARIS_LOOP_BLOCK(r->rel_.){
				double pmI[16], pmJ[16];
				s_pm_dot_pm(b->is_I_ ? r->i_diag_->pm_ : r->j_diag_->pm_, *b->cst_->makI()->prtPm(), pmI);
				s_pm_dot_pm(b->is_I_ ? r->j_diag_->pm_ : r->i_diag_->pm_, *b->cst_->makJ()->prtPm(), pmJ);

				// Calculate the positional error represented by cp of the constraint, store it in bc_. Only used in kinPos()
				if (cpt_cp) {
					if(auto j = dynamic_cast<const aris::dynamic::Joint*>(b->cst_))
						j->cptCpFromPm(r->bc_ + pos, pmI, pmJ);// cp //
					else
						dynamic_cast<const aris::dynamic::MotionBase*>(b->cst_)->cptCpFromPm(r->bc_ + pos, pmI, pmJ, pd_->active_mp_ + b->mot_mp_pos_);// cp //
				}

				double cmI[36], cmJ[36];
				b->cst_->cptGlbCmFromPm(cmI, cmJ, pmI, pmJ);
				s_mc(6, b->cst_->dim(), cmI, b->cst_->dim(), r->cmI_ + pos, r->rel_.size_);
				s_mc(6, b->cst_->dim(), cmJ, b->cst_->dim(), r->cmJ_ + pos, r->rel_.size_);
				pos += b->cst_->dim();
			}

			if (cpt_cp)for (Size i{ 0 }; i < r->rel_.size_; ++i)error_ = std::max(error_, std::abs(r->bc_[i])); // error //
		}
	}
	auto SubSystem::updDiagIv()noexcept->void { ARIS_LOOP_D s_iv2iv(*d->part_->pm(), d->part_->prtIv(), d->iv_); }
	auto SubSystem::updCv()noexcept->void{
		// bc in diag //
		ARIS_LOOP_D_2_TO_END{
			Size pos{ 0 };
			ARIS_LOOP_BLOCK(d->rel_.){
				b->cst_->cptCvDiff(d->bc_ + pos);
				pos += b->cst_->dim();
			}
		}
		// bc in remainder //
		ARIS_LOOP_R{
			Size pos{ 0 };
			ARIS_LOOP_BLOCK(r->rel_.){
				b->cst_->cptCvDiff(r->bc_ + pos);
				pos += b->cst_->dim();
			}
		}
	}
	auto SubSystem::updCa()noexcept->void{
		// bc in diag //
		ARIS_LOOP_D_2_TO_END{
			Size pos{ 0 };
			ARIS_LOOP_BLOCK(d->rel_.){
				b->cst_->cptCa(d->bc_ + pos);
				pos += b->cst_->dim();
			}
		}
		// bc in remainder //
		ARIS_LOOP_R{
			Size pos{ 0 };
			ARIS_LOOP_BLOCK(r->rel_.){
				b->cst_->cptCa(r->bc_ + pos);
				pos += b->cst_->dim();
			}
		}
	}
	auto SubSystem::updF()noexcept->void{
		// check F size //
		s_fill(fm_, fn_, 0.0, pd_->F_);

		Size cols{ 0 };
		ARIS_LOOP_R{
			ARIS_LOOP_BLOCK(r->){
				s_mm(6 - b->diag_->rel_.dim_, r->rel_.size_, 6, b->diag_->dm_ + at(b->diag_->rel_.dim_, 0, 6), 6, b->is_I_ ? r->cmI_ : r->cmJ_, r->rel_.size_, pd_->F_ + at(b->diag_->rows_, cols, ColMajor(fm_)), ColMajor(fm_));
			}
			cols += r->rel_.size_;
		}

		s_householder_utp(fm_, fn_, pd_->F_, ColMajor(fm_), pd_->FU_, ColMajor(fm_), pd_->FT_, 1, pd_->FP_, fr_, max_error_);
	}
	auto SubSystem::sovXp()noexcept->void{
		// Please refer to step 4, pre-update and initialize xp here //
		std::fill_n(d_data_[0].xp_, 6, 0.0);
		ARIS_LOOP_D_2_TO_END{
			// If there are multiple parts, resorting is needed //
			s_permutate(d->rel_.size_, 1, d->p_, d->bc_);

			// pre-update //
			s_mm(6, 1, d->rel_.dim_, d->dm_, T(6), d->bc_, 1, d->xp_, 1);
		}

		// Construct bcf, the formula F' * ypf = bcf exists in Step5 //
		Size cols{ 0 };
		ARIS_LOOP_R{
			s_vc(r->rel_.size_, r->bc_, pd_->bcf_ + cols);
			ARIS_LOOP_BLOCK(r->){
				auto cm = b->is_I_ ? r->cmJ_ : r->cmI_;// It is inverted here, because adding to the right needs to multiply by -1.0
				s_mma(r->rel_.size_, 1, 6, cm, ColMajor{ r->rel_.size_ }, b->diag_->xp_, 1, pd_->bcf_ + cols, 1);// Please refer to step 4, extract the pre-updated elements
			}
			cols += r->rel_.size_;
		}

		// Solve F' * xpf = bcf //
		s_vc(fn_, pd_->bcf_, pd_->xpf_);
		s_permutate(fn_, 1, pd_->FP_, pd_->xpf_);
		s_sov_lm(fr_, 1, pd_->FU_, T(ColMajor(fm_)), pd_->xpf_, 1, pd_->xpf_, 1, max_error_);
		s_householder_ut_q_dot(fm_, fn_, 1, pd_->FU_, ColMajor(fm_), pd_->FT_, 1, pd_->xpf_, 1, pd_->xpf_, 1);

		// Update xp //  Equivalent to D' * yp
		ARIS_LOOP_D_2_TO_END s_mma(6, 1, 6 - d->rel_.dim_, d->dm_ + at(0, d->rel_.dim_, T(6)), T(6), pd_->xpf_ + d->rows_, 1, d->xp_, 1);

		// Perform row transformation //  Equivalent to P' * (D' * yp)
		ARIS_LOOP_D_2_TO_END s_va(6, d->rd_->xp_, d->xp_);
	}
	auto SubSystem::updG()noexcept->void{
		gm_ = hasGround() ? fm_ : fm_ + 6;
		gn_ = hasGround() ? fm_ - fr_ : fm_ - fr_ + 6;

		//////////////////////////////////////////// Solve S for CT * xp = bc step 5 //////////////////////////////
		std::fill_n(pd_->xpf_, fm_, 0.0);
		for (Size j(-1); ++j < fm_ - fr_;){
			pd_->xpf_[fr_ + j] = 1.0;
			s_householder_ut_q_dot(fm_, fn_, 1, pd_->FU_, ColMajor(fm_), pd_->FT_, 1, pd_->xpf_, 1, pd_->S_ + j, fm_ - fr_);
			pd_->xpf_[fr_ + j] = 0.0;
		}

		//////////////////////////////////////////// Solve G, reference step 6 //////////////////////////////
		s_fill(gm_, gn_, 0.0, pd_->G_);
		// First solve the G generated by S
		for (Size j(-1); ++j < fm_ - fr_;){
			// Initialize xp and multiply by DT
			std::fill(d_data_[0].xp_, d_data_[0].xp_ + 6, 0.0);
			ARIS_LOOP_D_2_TO_END{
				if (d->rel_.dim_ == 6)std::fill(d->xp_, d->xp_ + 6, 0.0);
				else s_mm(6, 1, 6 - d->rel_.dim_, d->dm_ + at(0, d->rel_.dim_, ColMajor{ 6 }), ColMajor{ 6 }, pd_->S_ + at(d->rows_, j, fm_ - fr_), fm_ - fr_, d->xp_, 1);
			}

			// Multiply by PT, add rows
			ARIS_LOOP_D_2_TO_END s_va(6, d->rd_->xp_, d->xp_);

			// Multiply by I
			ARIS_LOOP_D_2_TO_END{
				double tem[6];
				s_iv_dot_as(d->iv_, d->xp_, tem);
				s_vc(6, tem, d->xp_);
			}

			// Multiply by P, add rows
			ARIS_LOOP_DIAG_INVERSE_2_TO_END s_va(6, d->xp_, d->rd_->xp_);

			// Multiply by D, and extract
			ARIS_LOOP_DIAG_INVERSE_2_TO_END s_mm(6 - d->rel_.dim_, 1, 6, d->dm_ + at(d->rel_.dim_, 0, 6), 6, d->xp_, 1, pd_->G_ + at(d->rows_, j, gn_), gn_);

			// If no ground, need to consider xp of the first part in G
			if (!hasGround())s_vc(6, d_data_->xp_, 1, pd_->G_ + at(fm_, j, fm_ - fr_ + 6), fm_ - fr_ + 6);
		}
		// Second, solve the G generated by the first part without ground
		if (!hasGround()){
			for (Size j(-1); ++j < 6;){
				// Initialize, here no need to multiply by DT, as first part's DT is identity matrix, and other xp are 0
				ARIS_LOOP_D std::fill(d->xp_, d->xp_ + 6, 0.0);
				d_data_[0].xp_[j] = 1.0;

				// Multiply by PT
				ARIS_LOOP_D_2_TO_END s_va(6, d->rd_->xp_, d->xp_);

				// Multiply by I, because bp stores external forces, bp cannot be used
				ARIS_LOOP_D	{
					double tem[6];
					s_iv_dot_as(d->iv_, d->xp_, tem);
					s_vc(6, tem, d->xp_);
				}

				// Multiply by P, similar to rowAddBp();
				ARIS_LOOP_DIAG_INVERSE_2_TO_END s_va(6, d->xp_, d->rd_->xp_);

				// Multiply by D, and extract
				ARIS_LOOP_DIAG_INVERSE_2_TO_END s_mm(6 - d->rel_.dim_, 1, 6, d->dm_ + at(d->rel_.dim_, 0, 6), 6, d->xp_, 1, pd_->G_ + at(d->rows_, fm_ - fr_ + j, gn_), gn_);
				s_vc(6, d_data_[0].xp_, 1, pd_->G_ + at(fm_, fm_ - fr_ + j, gn_), gn_);
			}
		}
	}
	auto SubSystem::sovXc()noexcept->void{
		/////////////////////////////////// Solve beta /////////////////////////////////////////////////////////////
		//// Update the force pf of each part ////
		ARIS_LOOP_D	{
			// External force (excluding inertial force) is already stored in bp, but for possible later corrections, it and the inertial force are temporarily stored in last_pm_ //
			
			// v x I * v //
			double I_dot_v[6];
			s_iv_dot_as(d->iv_, d->part_->vs(), I_dot_v);
			s_cfa(d->part_->vs(), I_dot_v, d->bp_);
			s_vc(6, d->bp_, d->last_pm_); // Temporary storage processing

			// I*(a-g) //
			double as_minus_g[6], iv_dot_as[6];
			s_vc(6, d->xp_, as_minus_g);// xp stores acceleration
			s_vs(6, pd_->gravity_, as_minus_g);
			s_iv_dot_as(d->iv_, as_minus_g, iv_dot_as);
			s_va(6, iv_dot_as, d->bp_);
		}

		//// P*bp  Apply row addition transformation to bp and extract bcf ////
		ARIS_LOOP_DIAG_INVERSE_2_TO_END {
			// Row transformation
			s_va(6, d->bp_, d->rd_->bp_);

			// Extract bcf
			double tem[6];
			s_mm(6, 1, 6, d->dm_, 6, d->bp_, 1, tem, 1);
			s_vc(6, tem, d->bp_);
			s_vc(6 - d->rel_.dim_, d->bp_ + d->rel_.dim_, pd_->bpf_ + d->rows_);
		}

		//// Solve beta in S solution space according to G, first make all beta as right side unknown variables ////
		if (!hasGround())s_vc(6, d_data_[0].bp_, pd_->beta_ + fm_);//This item actually copies xp without ground into beta
		s_householder_ut_qt_dot(fm_, fr_, 1, pd_->FU_, ColMajor(fm_), pd_->FT_, 1, pd_->bpf_, 1, pd_->beta_, 1);

		//// Calculate QT_DOT_G ////
		// G and QT_DOT_G are in the same memory, so the following first sentence is not needed
		// if (!hasGround())s_mc(gm - fm_, gn_, G + at(fm_, 0, gn_), QT_DOT_G + at(fm_, 0, gn_)); // This item actually copies the G generated without ground into QT_DOT_G
		s_householder_ut_qt_dot(fm_, fr_, gn_, pd_->FU_, ColMajor(fm_), pd_->FT_, 1, pd_->G_, gn_, pd_->QT_DOT_G_, gn_);

		///////////////////////////////
		// Solve the coefficient beta of the previous general solution
		// It is possible to judge whether the particle etc. affect the calculation by rank == m-r
		///////////////////////////////
		Size rank;
		s_householder_utp(gn_, gn_, pd_->QT_DOT_G_ + at(fr_, 0, gn_), pd_->GU_ + at(fr_, 0, gn_), pd_->GT_, pd_->GP_, rank, max_error_);
		s_householder_utp_sov(gn_, gn_, 1, rank, pd_->GU_ + at(fr_, 0, gn_), pd_->GT_, pd_->GP_, pd_->beta_ + fr_, pd_->beta_);

		/////////////////////////////////// Solve xp /////////////////////////////////////////////////////////////
		//// Recalculate xp, considering inertia this time ////
		// Update xpf according to the particular solution and the particular solution without ground (velocity of part 1 can be set arbitrarily before)
		s_mms(fm_, 1, fm_ - fr_, pd_->S_, pd_->beta_, pd_->xpf_);
		if (!hasGround())s_vi(6, pd_->beta_ + fm_ - fr_, d_data_[0].xp_);
		// Update xpf to xp, multiply by D\' then multiply by P\'
		ARIS_LOOP_D_2_TO_END {
			// Combine with bc and multiply by D\'
			s_mm(6, 1, d->rel_.dim_, d->dm_, ColMajor{ 6 }, d->bc_, 1, d->xp_, 1);
			s_mma(6, 1, 6 - d->rel_.dim_, d->dm_ + at(0, d->rel_.dim_, T(6)), T(6), pd_->xpf_ + d->rows_, 1, d->xp_, 1);

			// Multiply by P\'
			s_va(6, d->rd_->xp_, d->xp_);
		}

		/////////////////////////////////// Solve xc /////////////////////////////////////////////////////////////
		// Since xp above may not be the true solution, recalculate cyclically here
		//// Update the force pf of each part ////
		ARIS_LOOP_D {
			// Extract previously temporarily stored external force //
			s_vc(6, d->last_pm_, d->bp_);

			// I*(a-g) //
			double as_minus_g[6], iv_dot_as[6];
			s_vc(6, d->xp_, as_minus_g);// xp stores acceleration
			s_vs(6, pd_->gravity_, as_minus_g);
			s_iv_dot_as(d->iv_, as_minus_g, iv_dot_as);
			s_va(6, iv_dot_as, d->bp_);
		}

		//// P*bp  Apply row addition transformation to bp ////
		ARIS_LOOP_DIAG_INVERSE_2_TO_END {
			s_va(6, d->bp_, d->rd_->bp_);
			
			double tem[6];
			s_mm(6, 1, 6, d->dm_, 6, d->bp_, 1, tem, 1);
			s_vc(6, tem, d->bp_);
			s_vc(6 - d->rel_.dim_, d->bp_ + d->rel_.dim_, pd_->bpf_ + d->rows_);
		}
		
		// Solve xcf //
		s_householder_utp_sov(fm_, fn_, 1, fr_, pd_->FU_, ColMajor(fm_), pd_->FT_, 1, pd_->FP_, pd_->bpf_, 1, pd_->xcf_, 1, max_error_);

		// Update the calculated x to remainder, then move the known variable to the right side
		Size cols{ 0 };
		ARIS_LOOP_R {
			s_vc(r->rel_.size_, pd_->xcf_ + cols, r->xc_);

			// Update to be solved
			ARIS_LOOP_BLOCK(r->){
				double tem[6];
				s_mm(6, 1, r->rel_.size_, b->is_I_ ? r->cmJ_ : r->cmI_, pd_->xcf_ + cols, tem);
				s_mma(6, 1, 6, b->diag_->dm_, 6, tem, 1, b->diag_->bp_, 1);
			}

			cols += r->rel_.size_;
		}
		ARIS_LOOP_D_2_TO_END {
			s_vc(d->rel_.dim_, d->bp_, d->xc_);
			std::fill(d->xc_ + d->rel_.dim_, d->xc_ + d->rel_.size_, 0.0);
			s_permutate_inv(d->rel_.size_, 1, d->p_, d->xc_);
		}
	}
	auto SubSystem::sovProjectMassMatrix()noexcept->void{
		/////////////////////////////////// Solve beta /////////////////////////////////////////////////////////////
		//// Update the force pf of each part ////
		ARIS_LOOP_D	{
			// External force (excluding inertial force) is already stored in bp, but for possible later corrections, it and the inertial force are temporarily stored in last_pm_ //
			
			// v x I * v //
			double I_dot_v[6];
			s_iv_dot_as(d->iv_, d->part_->vs(), I_dot_v);
			s_cfa(d->part_->vs(), I_dot_v, d->bp_);
			s_vc(6, d->bp_, d->last_pm_); // Temporary storage processing

			// I*(a-g) //
			double as_minus_g[6], iv_dot_as[6];
			s_vc(6, d->xp_, as_minus_g);// xp stores acceleration
			s_vs(6, pd_->gravity_, as_minus_g);
			s_iv_dot_as(d->iv_, as_minus_g, iv_dot_as);
			s_va(6, iv_dot_as, d->bp_);
		}

		//// P*bp  Apply row addition transformation to bp and extract bcf ////
		ARIS_LOOP_DIAG_INVERSE_2_TO_END {
			// Row transformation
			s_va(6, d->bp_, d->rd_->bp_);

			// Extract bcf
			double tem[6];
			s_mm(6, 1, 6, d->dm_, 6, d->bp_, 1, tem, 1);
			s_vc(6, tem, d->bp_);
			s_vc(6 - d->rel_.dim_, d->bp_ + d->rel_.dim_, pd_->bpf_ + d->rows_);
		}

		//// Solve beta in S solution space according to G, first make all beta as right side unknown variables ////
		if (!hasGround())s_vc(6, d_data_[0].bp_, pd_->beta_ + fm_);//This item actually copies xp without ground into beta
		s_householder_ut_qt_dot(fm_, fr_, 1, pd_->FU_, ColMajor(fm_), pd_->FT_, 1, pd_->bpf_, 1, pd_->beta_, 1);

		//// Calculate QT_DOT_G ////
		// G and QT_DOT_G are in the same memory, so the following first sentence is not needed
		// if (!hasGround())s_mc(gm - fm_, gn_, G + at(fm_, 0, gn_), QT_DOT_G + at(fm_, 0, gn_)); // This item actually copies the G generated without ground into QT_DOT_G
		s_householder_ut_qt_dot(fm_, fr_, gn_, pd_->FU_, ColMajor(fm_), pd_->FT_, 1, pd_->G_, gn_, pd_->QT_DOT_G_, gn_);
	}
	auto SubSystem::sovXcRemain()noexcept->void{
		///////////////////////////////
		// Solve the coefficient beta of the previous general solution
		// It is possible to judge whether the particle etc. affect the calculation by rank == m-r
		///////////////////////////////
		Size rank;
		s_householder_utp(gn_, gn_, pd_->QT_DOT_G_ + at(fr_, 0, gn_), pd_->GU_ + at(fr_, 0, gn_), pd_->GT_, pd_->GP_, rank, max_error_);
		s_householder_utp_sov(gn_, gn_, 1, rank, pd_->GU_ + at(fr_, 0, gn_), pd_->GT_, pd_->GP_, pd_->beta_ + fr_, pd_->beta_);

		/////////////////////////////////// Solve xp /////////////////////////////////////////////////////////////
		//// Recalculate xp, considering inertia this time ////
		// Update xpf according to the particular solution and the particular solution without ground (velocity of part 1 can be set arbitrarily before)
		s_mms(fm_, 1, fm_ - fr_, pd_->S_, pd_->beta_, pd_->xpf_);
		if (!hasGround())s_vi(6, pd_->beta_ + fm_ - fr_, d_data_[0].xp_);
		// Update xpf to xp, multiply by D\' then multiply by P\'
		ARIS_LOOP_D_2_TO_END {
			// Combine with bc and multiply by D\'
			s_mm(6, 1, d->rel_.dim_, d->dm_, ColMajor{ 6 }, d->bc_, 1, d->xp_, 1);
			s_mma(6, 1, 6 - d->rel_.dim_, d->dm_ + at(0, d->rel_.dim_, T(6)), T(6), pd_->xpf_ + d->rows_, 1, d->xp_, 1);

			// Multiply by P\'
			s_va(6, d->rd_->xp_, d->xp_);
		}

		/////////////////////////////////// Solve xc /////////////////////////////////////////////////////////////
		// Since xp above may not be the true solution, recalculate cyclically here
		//// Update the force pf of each part ////
		ARIS_LOOP_D {
			// Extract previously temporarily stored external force //
			s_vc(6, d->last_pm_, d->bp_);

			// I*(a-g) //
			double as_minus_g[6], iv_dot_as[6];
			s_vc(6, d->xp_, as_minus_g);// xp stores acceleration
			s_vs(6, pd_->gravity_, as_minus_g);
			s_iv_dot_as(d->iv_, as_minus_g, iv_dot_as);
			s_va(6, iv_dot_as, d->bp_);
		}

		//// P*bp  Apply row addition transformation to bp ////
		ARIS_LOOP_DIAG_INVERSE_2_TO_END {
			s_va(6, d->bp_, d->rd_->bp_);
			
			double tem[6];
			s_mm(6, 1, 6, d->dm_, 6, d->bp_, 1, tem, 1);
			s_vc(6, tem, d->bp_);
			s_vc(6 - d->rel_.dim_, d->bp_ + d->rel_.dim_, pd_->bpf_ + d->rows_);
		}
		
		// Solve xcf //
		s_householder_utp_sov(fm_, fn_, 1, fr_, pd_->FU_, ColMajor(fm_), pd_->FT_, 1, pd_->FP_, pd_->bpf_, 1, pd_->xcf_, 1, max_error_);

		// Update the calculated x to remainder, then move the known variable to the right side
		Size cols{ 0 };
		ARIS_LOOP_R {
			s_vc(r->rel_.size_, pd_->xcf_ + cols, r->xc_);

			// Update to be solved
			ARIS_LOOP_BLOCK(r->){
				double tem[6];
				s_mm(6, 1, r->rel_.size_, b->is_I_ ? r->cmJ_ : r->cmI_, pd_->xcf_ + cols, tem);
				s_mma(6, 1, 6, b->diag_->dm_, 6, tem, 1, b->diag_->bp_, 1);
			}

			cols += r->rel_.size_;
		}
		ARIS_LOOP_D_2_TO_END {
			s_vc(d->rel_.dim_, d->bp_, d->xc_);
			std::fill(d->xc_ + d->rel_.dim_, d->xc_ + d->rel_.size_, 0.0);
			s_permutate_inv(d->rel_.size_, 1, d->p_, d->xc_);
		}
	}
	
	auto SubSystem::cptJacobiBeta(std::vector<Size>& targetBodies, 
		std::vector<double>& target_xp) noexcept -> void {
    const bool grounded = hasGround();
    const int n_beta = grounded ? (fm_ - fr_) : (fm_ + 6 - fr_);
    const int n_body6 = 6 * d_size_;   // Rows = number of bodies * 6

    // Save current velocity state (for recovery)
    std::vector<double> save_xp(n_body6, 0.0);
    for (int b = 0; b < d_size_; ++b)
      s_vc(6, d_data_[b].xp_, save_xp.data() + 6 * b);

    // Clear all body velocities and constraint velocity errors (bc)
    // ARIS_LOOP_R std::fill_n(r->bc_, r->rel_.size_, 0.0);

    // Temporary array: length of xpf fm_
    std::vector<double> xpf(fm_, 0.0);

    // Column-by-column perturbation β
    for (int j = 0; j < n_beta; ++j) {
      // Before each column starts, clear all body velocities (baseline is zero)
    	ARIS_LOOP_D std::fill_n(d->xp_, 6, 0.0);

      if (j < fm_ - fr_) {
        // Internal motion DoF (via matrix S)
        // Construct perturbation vector of beta = e_j (length fm_ - fr_)
        std::vector<double> beta_pert(fm_ - fr_, 0.0);
        beta_pert[j] = 1.0;

        // xpf = S * beta_pert
        // Note: pd_->S_ is stored in column-major order, dimension fm_ × (fm_-fr_)
        s_mms(fm_, 1, fm_ - fr_, pd_->S_, beta_pert.data(), xpf.data());

        // Convert xpf to velocities of parts (base part keeps 0)
				ARIS_LOOP_D_2_TO_END{
          // D' * xpf_sub
          s_mma(6, 1, 6 - d->rel_.dim_,
                d->dm_ + at(0, d->rel_.dim_, T(6)), T(6),
                xpf.data() + d->rows_, 1,
                d->xp_, 1);
          // P' Row transformation (forward propagation)
          s_va(6, d->rd_->xp_, d->xp_);
        }
      } else {
        // Base motion DoF (6-dimensional) when there is no ground
        int base_idx = j - (fm_ - fr_);   // 0..5

        // Directly set the base body velocity to the unit vector e_{base_idx}
        d_data_[0].xp_[base_idx] = 1.0;

        // Forward kinematics propagates base velocity to all dependent bodies
				ARIS_LOOP_D_2_TO_END{
          // rd_->xp_ wraps the velocity transformation from parent to child
          s_va(6, d->rd_->xp_, d->xp_);
        }
      }

      // Extract target body velocity
      for (Size t = 0; t < targetBodies.size(); ++t) {
        int b = targetBodies[t];
        std::copy_n(d_data_[b].xp_, 6, target_xp.data() + 6*t);
      }
    }

    // Restore velocity state
    for (int b = 0; b < d_size_; ++b)
      s_vc(6, save_xp.data() + 6 * b, d_data_[b].xp_);
	}
	// The exact solution thinks that for a robot\'s exact solution, those restricted DoFs will not be modified, so when two different
	// FixedJoints of locations are added, because there is no DoF to adjust, the position of parts will not be adjusted
	// If there is a requirement to still adjust according to least squares even when there are no DoFs, then this method cannot be used
	// It depends on the credibility of the newly added constraint, if it is considered to have the same credibility as the robot connection, then a
	// global least squares is required, but because our contact constraints are not true rigid constraints, it is only necessary to have a
	// least squares solution in joint space, rather than a least squares solution that truly integrates the contact position.
	auto SubSystem::kinPos()noexcept->void{
		updDmCm(true);
		for (iter_count_ = 0; iter_count_ < max_iter_count_; ++iter_count_){
			if (error_ < max_error_) return;

			// solve
			updF();
			sovXp();

			// Update xp to pm
			ARIS_LOOP_D {
				std::swap(d->pm_, d->last_pm_);
				double tem[16];
				s_ps2pm(d->xp_, tem);
				// last_pm_ * xp_ -> pm_ solves the positional variation (least squares)
				s_pm2pm(tem, d->last_pm_, d->pm_);
			}

			double last_error = error_;
			updDmCm(true);

			// For non-serial arms, when the iteration error increases instead, the step size will be actively reduced
			// d_size_ = prt_vec.size()
			// Only non-serial arms will use the following iteration, sys.r_size_ = r_vec.size();
			// Only non-serial manipulator will have non-zero r_size_ = rel_vec.size() - prt_vec.size() + 1
			if (r_size_){
				// If while is used here, it can ensure that the error of each loop will decrease, but it may cause an infinite loop
				if (error_ > last_error) {
				
					double coe = last_error / (error_ + last_error);

					ARIS_LOOP_D	{
						s_nv(6, coe, d->xp_);
						double tem[16];
						s_ps2pm(d->xp_, tem);
						s_pm2pm(tem, d->last_pm_, d->pm_);
					}

					updDmCm(true);
				}
			}
		}
	}
	auto SubSystem::kinVel()noexcept->void{
		// make b
		updCv();

		// make A
		updDmCm(false);

		// solve
		updF();
		sovXp();
	}
	auto SubSystem::dynAccAndFce()noexcept->void{
		// upd Iv dm cm and ca  //
		updDiagIv();
		updDmCm(false);
		updCa();

		// upd F and G //
		updF();
		updG();

		// Solve a certain particular solution of xp (ignoring inertia), solve beta and xc //
		sovXp();
		sovXc();
	}
#undef ARIS_LOOP_D
#undef ARIS_LOOP_D_2_TO_END
#undef ARIS_LOOP_DIAG_INVERSE_2_TO_END
#undef ARIS_LOOP_R
#define ARIS_LOOP_SYS for (auto sys = imp_->pd_->subsys_data_; sys < imp_->pd_->subsys_data_ + imp_->pd_->subsys_size_; ++sys)
#define ARIS_LOOP_SYS_D for (auto d = sys->d_data_; d < sys->d_data_ + sys->d_size_; ++d)
#define ARIS_LOOP_SYS_R for (auto r = sys->r_data_; r < sys->r_data_ + sys->r_size_; ++r)
	struct UniversalSolver::Imp{
		// Dynamics calculates the relationships of the following variables
		// I: Inertia matrix, m x m
		// C: Constraint matrix, m x n
		// pa: Spatial acceleration of parts m x 1
		// pf: Spatial external force of parts (excluding inertial force) m x 1
		// ca: Acceleration of constraints (not spatial) n x 1
		// cf: Constraint force n x 1
		// Dynamics mainly solves the following equations:
		// [ -I  C  ]  *  [ pa ]  = [ pf ]
		// [  C' O  ]     [ cf ]    [ ca ]
		//
		// A = [-I  C ]
		//     [ C' O ]
		//
		// x = [ pa ] = [ xp ]
		//     [ cf ]   [ xc ]
		//
		// b = [ pf ] = [ bp ]
		//     [ ca ]   [ bc ]
		// 
		// 
		// Calculation method:
		// --------------------------------------------------------------------
		// step 1: Separate subsystems, adjust the order of parts and constraints so that the constraint matrix becomes an upper triangular block matrix
		//        After this step, the constraint matrix becomes:
		//
		//        With ground (n constraints, m parts):
		//        [ Cg  -C1  ...              ... -Cn ]
		//        |      C1  ...  -Cm-1       ...     |
		//        |          ...         -Cm  ...     |
		//        [                Cm-1   Cm  ...  Cn ]                
		//    
		//        Without ground (n constraints, m parts):
		//        [ -C1  ...             -Cn ]
		//        |  C1  ...  -Cm-1          |
		//        |      ...         -Cm     |
		//        [            Cm-1   Cm  Cn ]  
		// --------------------------------------------------------------------
		// step 2: Find matrix P to make the constraint matrix diagonalized
		//        
		//        With ground:
		//        P * C = [ Cg                 -Cm ... -Cn ]
		//                |    C1              -Cm ...     |
		//                |       C2               ...  Cn | 
		//                |          ...        Cm ...  Cn |
		//                [              Cm-1      ...  Cn ]
		//        Without ground, at this time the first row is empty:
		//        P * C = [                                ]
		//                |    C1              -Cm ...     |
		//                |       C2               ...  Cn | 
		//                |          ...        Cm ...  Cn |
		//                [              Cm-1      ...  Cn ]
		// --------------------------------------------------------------------
		// step 3: Multiply both sides by matrix D, and a sub-block F can be obtained
		//        The definition of Matrix D is: D1_6x6 * C1_6xn = I_6xn where n is the dimension of constraints
		// 
		//        With ground:
		//                                                              cm            cn
		//        D * P * C = [ I                                      -Cm  ...      -Cn ]
		//                    |    [ I1 ]                           -D1*Cm  ...          |
		//                    |    [  0 ]                                                |  r2
		//                    |            [ I2 ]                           ...    D2*Cn |  
		//                    |            [  0 ]                                        |  r3
		//                    |                    ...               Di*Cm  ...    Di*Cn |
		//                    |                         [ Im-1 ]                         |
		//                    [                         [   0  ]            ...  Dm-1*Cn ]  rm
		//
		//        Without ground, at this time the first row is empty:
		//                                                           cm            cn
		//        D * P * C = [                                                       ]
		//                    | [ I1 ]                           -D1*Cm  ...          |
		//                    | [  0 ]                                                |  r2
		//                    |         [ I2 ]                           ...    D2*Cn |
		//                    |         [  0 ]                                        |  r3
		//                    |                 ...               Di*Cm  ...    Di*Cn |
		//                    |                      [ Im-1 ]                         |
		//                    [                      [   0  ]            ...  Dm-1*Cn ]  rm
		//
		//        Extract cols cm ... cn, rows r2 ... rm, to form F
		//        F = D * P * C  (r2 ... rm , cm ... cn)
		// --------------------------------------------------------------------
		// step 4: Solve the equation C\' * xp = bc
		//        This equation can be formulated as:
		//           C' * P' * D' * D'^-1 * P'^-1 * xp = bc
		//           DPC' * D'^-1 * P'^-1 * xp = bc
		//           DPC' * yp = bc
		//        where: yp = D\'^-1 * P\'^-1 * xp
		//        
		//        DPC\' and yp are
		//        With ground:
		//                            r2        r3                  rm 
		//        DPC' = [  I                                            ]      [    yp1   ]
		//               |       [ I1 0 ]                                |      | -------- |
		//               |                 [ I2 0 ]                      |      | [ ypa2 ] |
		//               |                            ...                |      | [ ypf2 ] |
		//               |                                   [ Im-1 0 ]  |      |    ...   |
		//               | -Cm'  -Cm'*D1'           Cm'*Di'   Cm'*Dm-1'  |  c1  |    ...   |
		//               | ...      ...       ...   Cj'*Di'     ...      |      | [ ypam ] |
		//               [ -Cn'             Cn'*D2' Cn'*Di'   Cn'*Dm-1'  ]  cn  [ [ ypfm ] ]
		// 
		//        Without ground:
		//                            r2        r3                  rm 
		//        DPC' = [  0                                            ]
		//               |  0    [ I1 0 ]                                |
		//               |  0              [ I2 0 ]                      |  
		//               |  0                         ...                |
		//               |  0                                [ Im-1 0 ]  |  
		//               |  0    -Cm'*D1'           Cm'*Di'   Cm'*Dm-1'  |  c1
		//               |  0       ...       ...   Cj'*Di'     ...      | 
		//               [  0               Cn'*D2' Cn'*Di'   Cn'*Dm-1'  ]  cn
		// 
		//        yp is all the same:
		//        yp   = [    yp1   ]
		//               | -------- |
		//               | [ ypa2 ] |
		//               | [ ypf2 ] |
		//               | -------- |
		//               |    ...   |
		//               |    ...   |
		//               | -------- |
		//               | [ ypam ] |
		//               [ [ ypfm ] ]
		//        while bc is:
		//        bc   = [  bc1  ]
		//               |  bc2  |
		//               |  ...  |
		//               [  bcn  ]
		//
		//        thus a part of yp can be calculated:
		//        [ ypa2 ]  =  [  bc1  ]
		//        | ypa3 |     |  bc2  |
		//        |  ... |     |  ...  |
		//        [ ypam ]     [ bcm-1 ]
		//
		//        the other part of yp requires F mentioned above, where k below is the corresponding constraint dimension:
		//        F * [ ypf2 ]  =  [  bcm  ]   -   [ -Cm'  * D1(1:k,1:6)' * bc1 + ... + Cm'  * Dm-1(1:k,1:6)' * bcm-1 ]
		//            | ypf3 |     | bcm+1 |       |  Cm+1'* D1(1:k,1:6)' * bc1 + ... + Cm+1'* Dm-1(1:k,1:6)' * bcm-1 |
		//            |  ... |     |  ...  |       |                              ...                                 |
		//            [ ypfm ]     [  bcn  ]       [ -Cn'  * D1(1:k,1:6)' * bc1 + ... + Cn'  * Dm-1(1:k,1:6)' * bcm-1 ]
		// 
		//        the last part of yp is yp1, as follows:
		//        With ground:  yp1 = [0,0,0,0,0,0]'
		//        Without ground:  yp1 can take any value, its value cannot be determined in this section
		//        After determining yp, xp can be found
		//        xp = P' * D' * yp = P' * diag([yp1,  D1(1:k,1:6)'*bc1 + D1(k+1:6,1:6)'*ypf2], ... ,  Dm-1(1:k,1:6)'*bcm-1 + Dm-1(k+1:6,1:6)'*ypfm])
		//        
		//        In practical calculations, xp can first be set to D1(1:k,1:6)\'*bc1 to reduce computations
		// --------------------------------------------------------------------
		// step 5: Solve equation F\' * ypf = bcf for particular solution xpf and general solution S
		//        Here F\' is a matrix formed by extracting some elements from matrix F, since part of yp has already been found
		//
		//        F * P = Q * R
		//        Here R is: [R1 R2 | 0 0]
		//            1   r   n
		//        1 [ * * * * * ]
		//          |   * * * * |
		//        r |     * * * |
		//          |           |
		//          |           |
		//        m [           ]
		//
		//        Solving here:
		//        F' * ypf = bcf
		//        i.e.:
		//        P^-T * P' * F' * xpf = bcf
		//        P^-T * R' * Q' * xpf = bcf
		//        Here R\' is:
		//            1   r   m
		//        1 [ *         ]
		//          | * *       |
		//        r | * * *     |
		//          | * * *     |
		//          | * * *     |
		//        n [ * * *     ]
		//        Therefore the general solution of Q\' * x is:
		//            1  m-r 
		//        1 [       ]
		//          |       |
		//        r |       |
		//          | 1     |
		//          |   1   |
		//        m [     1 ]
		//        Therefore the general solution S of x is multiplying Q from the left above
		//        S = Q * [    0_rxr      ]
		//                [ I_(m-r)x(m-r) ]
		//
		//        P^-T * R' * Q' * xpf = bcf
		//        Its particular solution is:
		//        Q * [ R1^-1   ]  *  P' * bcf
		//            [       1 ]
		// --------------------------------------------------------------------
		// step 6: S is the general solution, now G needs to be determined to find the true constraint forces using the inertia matrix
		//        From S, the general solution of yp can be found:
		//        With ground:
		//        K1 = [ [   0   ] ]
		//             | ......... |
		//             | [   0   ] |
		//             | [  Sf1  ] |
		//             | ......... |  
		//             | [   0   ] |
		//             | [  Sf2  ] |
		//             |    ...    |
		//             | [   0   ] |
		//             [ [ Sfm-1 ] ]
		//        Without ground:
		//        K1 = [ I .         ]
		//             | ........... |
		//             |   . [  0  ] |
		//             |   . [ Sf1 ] |
		//             | ........... |
		//             |   . [  0  ] |
		//             |   . [ Sf2 ] |
		//             |   .   ...   |
		//             |   . [  0  ] |
		//             [   . [ Sfn ] ]
		//        
		//        Furthermore, the general solution K of xp can be determined:
		//        K = P' * D' * K1        
		//        
		//        Once K is found, the final equation becomes:
		//        [ I C ] * [ xpt + K*beta ] = [ bp ]
		//                  [     xc       ]
		//        Can be simplifed as:
		//        [ C I*K ] * [  xc  ] = bp - I * xpt
		//                    [ beta ]
		//        
		//        Multiply both sides by P*D, we get:
		//        [ PDC PDIK ] * [  xcf ] = PD(bp - I * xpt)
		//                       [ beta ]
		//        
		//        PDC and PDIK in cols cm ... cn : end, rows r2 ... rm, form [F G]
		//        With ground, F and G have identical number of rows, G\'s col count is fm - fr:
		//        [ F  G ] * [  xcf ] = bpf
		//                   [ beta ]
		//        Without ground, G\'s row count is fm + 6, col count is fm-fr+6:
		//        [ F  G1 ] * [  xcf ] = [ bpf ]
		//        [    G2 ]   [ beta ]   [ bp1 ]
		// --------------------------------------------------------------------
		// step 7: After finding G, the next step is to find beta
		//        Since F\'s QR decomposition was found, multiply by F\'s Q on both sides: Note, here G2 is not the previously mentioned ungrounded G2
		//        With ground:
		//                     fn    fm-fr
		//        fr       [ R*P^-1  [Q'*G](   1:fr,:)  ] * [  xcf ] = [Q'*bpf](   1:fr)
		//        fm-fr    [         [Q'*G](fr+1:fm,:)  ]   [ beta ]   [Q'*bpf](fr+1:fm)
		//        Without ground:
		//                     fn    fm-fr+6
		//        fr       [ R*P^-1  [Q'*G1](   1:fr,:)  ] * [  xcf  ] = [ [Q'*bpf](   1:fr) ]
		//        fm-fr    [         [Q'*G1](fr+1:fm,:)  ]   [ beta  ]   | [Q'*bpf](fr+1:fm) |
		//        6        [                G2           ]               [       bp1         ]
		//
		//        Solving the lower right corner gives beta
		//        Finally gives xcf
		//
		
		PublicData* pd_{ nullptr };
		std::vector<char> mem_pool_;

		static auto one_constraint_upd_d_and_cp(Diag *d, bool cpt_cp)noexcept->void{
			// Update pm //
			double pmI[16], pmJ[16];
			auto b = &d->rel_.blk_data_[0];
			s_pm_dot_pm(b->is_I_ ? d->pm_ : d->rd_->pm_, *b->cst_->makI()->prtPm(), pmI);
			s_pm_dot_pm(b->is_I_ ? d->rd_->pm_ : d->pm_, *b->cst_->makJ()->prtPm(), pmJ);
			
			// Calculate dm //
			d->rel_.blk_data_[0].cst_->cptGlbDmFromPm(d->dm_, pmI, pmJ);
			if (!d->rel_.blk_data_[0].is_I_)s_iv(36, d->dm_);
			
			// Calculate cp //
			if (cpt_cp) {
				if (auto mot = dynamic_cast<const aris::dynamic::MotionBase*>(d->rel_.blk_data_[0].cst_)) {
					mot->cptCpFromPm(d->bc_, pmI, pmJ, d->rel_.blk_data_[0].mp_);
				}
				else {
					dynamic_cast<const aris::dynamic::Joint*>(d->rel_.blk_data_[0].cst_)->cptCpFromPm(d->bc_, pmI, pmJ);
				}
			}
		}
		static auto revolute_upd_d_and_cp(Diag *d, bool cpt_cp)noexcept->void{
			// Update pm //
			double pmI[16], pmJ[16];
			auto b = &d->rel_.blk_data_[0];
			s_pm_dot_pm(b->is_I_ ? d->pm_ : d->rd_->pm_, *b->cst_->makI()->prtPm(), pmI);
			s_pm_dot_pm(b->is_I_ ? d->rd_->pm_ : d->pm_, *b->cst_->makJ()->prtPm(), pmJ);
			
			// Calculate dm //
			d->rel_.blk_data_[0].cst_->cptGlbDmFromPm(d->dm_, pmI, pmJ);
			if (!d->rel_.blk_data_[0].is_I_)s_iv(36, d->dm_);
			
			// Calculate cp //
			if (cpt_cp){
				auto m = static_cast<const Motion*>(d->rel_.blk_data_[1].cst_);
				
				double rm[9], pm_j_should_be[16];
				// Calculate rotation matrix rm based on true rotation around real z axis of revolute joint 
				s_rmz(m->mp2mpInternal(*d->rel_.blk_data_[1].mp_), rm);

				s_vc(16, pmJ, pm_j_should_be);
				// pmJ * rm -> pm_j_should_be Rotation matrices multiplication
				s_mm(3, 3, 3, pmJ, 4, rm, 3, pm_j_should_be, 4);

				double pm_j2i[16], ps_j2i[6];
				s_inv_pm_dot_pm(pmI, pm_j_should_be, pm_j2i);
				s_pm2ps(pm_j2i, ps_j2i);

				// The cp corresponding to motion is at the end //
				s_vc(m->axis(), ps_j2i, d->bc_);
				s_vc(5 - m->axis(), ps_j2i + m->axis() + 1, d->bc_ + m->axis());
				d->bc_[5] = ps_j2i[m->axis()];
			}
		}
		static auto prismatic_upd_d_and_cp(Diag *d, bool cpt_cp)noexcept->void{
			// Update pm //
			double pmI[16], pmJ[16];
			auto b = &d->rel_.blk_data_[0];
			s_pm_dot_pm(b->is_I_ ? d->pm_ : d->rd_->pm_, *b->cst_->makI()->prtPm(), pmI);
			s_pm_dot_pm(b->is_I_ ? d->rd_->pm_ : d->pm_, *b->cst_->makJ()->prtPm(), pmJ);

			// Calculate dm //
			d->rel_.blk_data_[0].cst_->cptGlbDmFromPm(d->dm_, pmI, pmJ);
			if (!d->rel_.blk_data_[0].is_I_)s_iv(36, d->dm_);
			
			// Calculate cp //
			if (cpt_cp)	{
				auto m = static_cast<const Motion*>(d->rel_.blk_data_[1].cst_);

				double pm_j_should_be[16];
				s_vc(16, pmJ, pm_j_should_be);
				s_va(3, m->mp2mpInternal(*d->rel_.blk_data_[1].mp_), pm_j_should_be + m->axis(), 4, pm_j_should_be + 3, 4);

				double pm_j2i[16], ps_j2i[6];
				s_inv_pm_dot_pm(pmI, pm_j_should_be, pm_j2i);
				s_pm2ps(pm_j2i, ps_j2i);

				// The cp corresponding to motion is at the end //
				s_vc(m->axis(), ps_j2i, d->bc_);
				s_vc(5 - m->axis(), ps_j2i + m->axis() + 1, d->bc_ + m->axis());
				d->bc_[5] = ps_j2i[m->axis()];
			}
		}
		// Generalized method for updating diag and cp for Joint and Motion constraints.
		static auto normal_upd_d_and_cp(Diag *d, bool cpt_cp)noexcept->void	{
			Size pos{ 0 };
			ARIS_LOOP_BLOCK(d->rel_.){
				// Update pm //
				double pmI[16], pmJ[16];
				s_pm_dot_pm(b->is_I_ ? d->pm_ : d->rd_->pm_, *b->cst_->makI()->prtPm(), pmI);
				s_pm_dot_pm(b->is_I_ ? d->rd_->pm_ : d->pm_, *b->cst_->makJ()->prtPm(), pmJ);

				// Calculate cp //
				if (cpt_cp) {
					if (auto mot = dynamic_cast<const aris::dynamic::MotionBase*>(b->cst_)) {
						mot->cptCpFromPm(d->bc_ + pos, pmI, pmJ, b->mp_);
					}
					else if(auto jnt = dynamic_cast<const aris::dynamic::Joint*>(b->cst_)) {
						jnt->cptCpFromPm(d->bc_ + pos, pmI, pmJ);
					}
				}

				// Calculate dm //
				double cmI_tem[36], cmJ_tem[36];
				b->cst_->cptGlbCmFromPm(cmI_tem, cmJ_tem, pmI, pmJ);
				// Update d->cmI_ and d->cmJ_ using the calculated Constraint matrix
				s_mc(6, b->cst_->dim(), cmI_tem, b->cst_->dim(), (b->is_I_ ? d->cmI_ : d->cmJ_) + pos, d->rel_.size_);
				s_mc(6, b->cst_->dim(), cmJ_tem, b->cst_->dim(), (b->is_I_ ? d->cmJ_ : d->cmI_) + pos, d->rel_.size_);
				pos += b->cst_->dim();
			}

			double Q[36];
			s_householder_utp(6, d->rel_.size_, d->cmI_, d->cmU_, d->cmT_, d->p_, d->rel_.dim_);
			s_householder_ut2qr(6, d->rel_.size_, d->cmU_, d->cmT_, Q, d->cmU_);

			double tem[36]{ 1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1 };
			s_inv_um(d->rel_.dim_, d->cmU_, d->rel_.size_, tem, 6);
			s_mm(6, 6, 6, tem, 6, Q, dynamic::ColMajor{ 6 }, d->dm_, 6);
		}
	};
	auto UniversalSolver::allocateMemory()->void{
		// for mem_pool
		Size mem_pool_size = 0;

		// Construct input mots, jnts, prts and lengths of related input and output variables
		int active_mot_size = 0,
			active_mp_size = 0,
			active_mot_dim = 0,
			deactive_mot_size = 0,
			deactive_mp_size = 0,
			deactive_mot_dim = 0;
		std::vector<aris::dynamic::MotionBase*> active_mot_vec, deactive_mot_vec;
		std::vector<const Part*> active_prt_vec;
		std::vector<Joint*> active_jnt_vec;
		{
			// Construct mots //
			for (auto& mot : model()->motionPool()) {
				if (mot.active()) {
					active_mot_size++;
					active_mp_size += (int)mot.pSize();
					active_mot_dim += (int)mot.dim();
					active_mot_vec.push_back(&mot);
				}
				else {
					deactive_mot_size++;
					deactive_mp_size += (int)mot.pSize();
					deactive_mot_dim += (int)mot.dim();
					deactive_mot_vec.push_back(&mot);
				}
			}
			for (auto& gmt : model()->generalMotionPool()) {
				if (gmt.active()) {
					active_mot_size++;
					active_mp_size += (int)gmt.pSize();
					active_mot_dim += (int)gmt.dim();
					active_mot_vec.push_back(&gmt);
				}
				else {
					deactive_mot_size++;
					deactive_mp_size += (int)gmt.pSize();
					deactive_mot_dim += (int)gmt.dim();
					deactive_mot_vec.push_back(&gmt);
				}
			}

			// Construct prts //
			active_prt_vec.push_back(&model()->ground());
			for (auto& p : model()->partPool())if (p.active() && &p != &model()->ground())active_prt_vec.push_back(&p);

			// Construct jnts //
			for (auto& jnt : model()->jointPool())if (jnt.active())active_jnt_vec.push_back(&jnt);
		}

		// Construct public variable block //
		PublicData pub_data;
		s_vc(6, model()->environment().gravity(), pub_data.gravity_);
		core::allocMem(mem_pool_size, imp_->pd_, 1);

		// Construct subsystems, first group prt and rel //
		std::vector<std::vector<const Part*>> prt_vec_vec;
		std::vector<std::vector<LocalRelation>> rel_vec_vec;
		{
			int mv_id = 0, mp_id = 0;

			// make active prt pool //
			auto active_part_pool = active_prt_vec;

			// make active constraint pool //
			std::vector<const Constraint*> cp;
			for (auto jnt : active_jnt_vec)cp.push_back(jnt);
			for (auto mot : active_mot_vec)cp.push_back(mot);

			// make relation pool //
			std::vector<LocalRelation> relation_pool;
			
			for (auto c : cp){
				if (c->makI() == nullptr || c->makJ() == nullptr) continue;
				
				auto ret = std::find_if(relation_pool.begin(), relation_pool.end(), [&c](auto &relation){
					auto ri{ relation.prtI_ }, rj{ relation.prtJ_ }, ci{ &c->makI()->fatherPart() }, cj{ &c->makJ()->fatherPart() };
					return ((ri == ci) && (rj == cj)) || ((ri == cj) && (rj == ci));
				});

				if (ret == relation_pool.end()){
					relation_pool.push_back(LocalRelation{ &c->makI()->fatherPart(), &c->makJ()->fatherPart(), c->dim(), c->dim() });
					relation_pool.back().cst_pool_.push_back({ c, true, dynamic_cast<const MotionBase*>(c) ? mv_id : -1, dynamic_cast<const MotionBase*>(c) ? mp_id : -1 });
					
					mv_id += dynamic_cast<const MotionBase*>(c) ? (int)dynamic_cast<const MotionBase*>(c)->vSize() : 0;
					mp_id += dynamic_cast<const MotionBase*>(c) ? (int)dynamic_cast<const MotionBase*>(c)->pSize() : 0;
				}
				else{
					ret->cst_pool_.push_back({ c, &c->makI()->fatherPart() == ret->prtI_, dynamic_cast<const MotionBase*>(c) ? mv_id : -1, dynamic_cast<const MotionBase*>(c) ? mp_id : -1 });
					std::sort(ret->cst_pool_.begin(), ret->cst_pool_.end(), [](auto& a, auto& b){return a.cst_->dim() > b.cst_->dim();});//Place larger constraints forward here
					ret->size_ += c->dim();
					ret->dim_ = ret->cst_pool_[0].cst_->dim();// relation dim follows the largest, the largest is first

					mv_id += dynamic_cast<const MotionBase*>(c) ? (int)dynamic_cast<const MotionBase*>(c)->vSize() : 0;
					mp_id += dynamic_cast<const MotionBase*>(c) ? (int)dynamic_cast<const MotionBase*>(c)->pSize() : 0;
				}
			}

			// Separate related Part and Relation //
			while (active_part_pool.size() > 1){
				std::function<void(std::vector<const Part *> &part_pool_, std::vector<const Part *> &left_part_pool, std::vector<LocalRelation> &relation_pool, const Part *part)> addPart;
				addPart = [&](std::vector<const Part *> &part_pool_, std::vector<const Part *> &left_part_pool, std::vector<LocalRelation> &relation_pool, const Part *part)->void	{
					if (std::find(part_pool_.begin(), part_pool_.end(), part) != part_pool_.end())return;

					part_pool_.push_back(part);

					// If not ground, wipe out the part, and add related parts of the part //
					if (part != left_part_pool.front())	{
						left_part_pool.erase(std::find(left_part_pool.begin(), left_part_pool.end(), part));
						for (auto &rel : relation_pool)	{
							if (rel.prtI_ == part || rel.prtJ_ == part)	{
								addPart(part_pool_, left_part_pool, relation_pool, rel.prtI_ == part ? rel.prtJ_ : rel.prtI_);
							}
						}
					}
				};

				// insert prt_vec and rel_vec //
				prt_vec_vec.push_back(std::vector<const Part*>());
				rel_vec_vec.push_back(std::vector<LocalRelation>());
				auto &prt_vec = prt_vec_vec.back();
				auto &rel_vec = rel_vec_vec.back();

				// add related part //
				addPart(prt_vec, active_part_pool, relation_pool, active_part_pool.at(1));

				// add related relation //
				for (auto &rel : relation_pool)	{
					if (std::find_if(prt_vec.begin(), prt_vec.end(), [&rel, this](const Part *prt) { return prt != &(this->model()->ground()) && (prt == rel.prtI_ || prt == rel.prtJ_); }) != prt_vec.end())
					{
						rel_vec.push_back(rel);
					}
				}

				// Sort parts and relations of sys //
				for (Size i = 0; i < std::min(prt_vec.size(), rel_vec.size()); ++i)	{
					// Sort parts first, find the next part linked to the previous part
					std::sort(prt_vec.begin() + i, prt_vec.end(), [i, this, &rel_vec](const Part* a, const Part* b)	{
						if (a == &this->model()->ground()) return true; // Ground has highest priority
						if (b == &this->model()->ground()) return false; // Ground has highest priority
						if (i == 0)return a->id() < b->id();// In the first round, find ground or other grounds first, to prevent the index i-1 below from failing
						if (b == rel_vec[i - 1].prtI_) return false;
						if (b == rel_vec[i - 1].prtJ_) return false;
						if (a == rel_vec[i - 1].prtI_) return true;
						if (a == rel_vec[i - 1].prtJ_) return true;
						return a->id() < b->id();
					});
					// Then insert relations connecting the new part
					std::sort(rel_vec.begin() + i, rel_vec.end(), [i, this, &prt_vec](Relation a, Relation b){
						auto pend = prt_vec.begin() + i + 1;
						auto a_part_i = std::find_if(prt_vec.begin(), pend, [a](const Part* p)->bool { return p == a.prtI_; });
						auto a_part_j = std::find_if(prt_vec.begin(), pend, [a](const Part* p)->bool { return p == a.prtJ_; });
						auto b_part_i = std::find_if(prt_vec.begin(), pend, [b](const Part* p)->bool { return p == b.prtI_; });
						auto b_part_j = std::find_if(prt_vec.begin(), pend, [b](const Part* p)->bool { return p == b.prtJ_; });

						bool a_is_ok = (a_part_i == pend) != (a_part_j == pend);
						bool b_is_ok = (b_part_i == pend) != (b_part_j == pend);

						if (a_is_ok && !b_is_ok) return true;
						else if (!a_is_ok && b_is_ok) return false;
						else if (a.size_ != b.size_)return a.size_ > b.size_;
						else if (a.dim_ != b.dim_)return a.dim_ > b.dim_;
						else return false;
					});
				}
			}
		}

		// Construct subsystem //
		std::vector<SubSystem> sys_vec;
		std::vector<std::vector<Diag>> d_vec_vec;
		std::vector<std::vector<LocalRemainder>> r_vec_vec;
		Size max_F_size{ 0 }, max_fm{ 0 }, max_fn{ 0 }, max_G_size{ 0 }, max_gm{ 0 }, max_gn{ 0 }, max_cm_size{ 0 };
		for (int i = 0; i < prt_vec_vec.size(); ++i){
			auto &prt_vec = prt_vec_vec[i];
			auto &rel_vec = rel_vec_vec[i];

			// Insert SubSystem //
			sys_vec.push_back(SubSystem());
			auto &sys = sys_vec.back();
			sys.max_error_ = maxError();
			sys.has_ground_ = (prt_vec.front() == &model()->ground());

			// Produce d_vec (diag_vec) //
			core::allocMem(mem_pool_size, sys.d_data_, prt_vec.size());
			d_vec_vec.push_back(std::vector<Diag>());
			auto &d_vec = d_vec_vec.back();
			d_vec.resize(prt_vec.size());
			d_vec[0].part_ = prt_vec[0];
			for (Size i = 1; i < d_vec.size(); ++i)	{
				auto &diag = d_vec[i];
				auto &rel = rel_vec[i - 1];

				// Change whether it is an I part based on diag
				if (rel.prtI_ != prt_vec.at(i))	{
					std::swap(rel.prtI_, rel.prtJ_);
					for (auto &c : rel.cst_pool_)c.is_I_ = !c.is_I_;
				}
				diag.part_ = prt_vec[i];

				// Allocate Relation::Block memory
				core::allocMem(mem_pool_size, rel.blk_data_, rel.cst_pool_.size());

				// Allocate size of p bc xc in Diag
				core::allocMem(mem_pool_size, d_vec[i].p_, rel.size_);
				core::allocMem(mem_pool_size, d_vec[i].bc_, rel.size_);
				core::allocMem(mem_pool_size, d_vec[i].xc_, rel.size_);

				// Calculate size of max_cm
				max_cm_size = std::max(max_cm_size, rel.size_);

				// Optimize the calculation of dm matrix below, because optimization changes calculation of system required memory, must be placed here //
				{
					if (rel.cst_pool_.size() == 1){ // Optimization for when there is only one constraint //
						diag.upd_d_and_cp_ = Imp::one_constraint_upd_d_and_cp;
					}
					// Optimization for revolute joint with rotary motor //
					else if (rel.cst_pool_.size() == 2
						&& dynamic_cast<const RevoluteJoint*>(rel.cst_pool_.at(0).cst_)
						&& dynamic_cast<const Motion*>(rel.cst_pool_.at(1).cst_)
						&& dynamic_cast<const Motion*>(rel.cst_pool_.at(1).cst_)->axis() == 5
						&& rel.cst_pool_.at(0).cst_->makI() == rel.cst_pool_.at(1).cst_->makI())
					{
						diag.upd_d_and_cp_ = Imp::revolute_upd_d_and_cp;
						rel.dim_ = 6;
					}
					// Optimization for prismatic joint with prismatic motor //
					else if (rel.cst_pool_.size() == 2
						&& dynamic_cast<const PrismaticJoint*>(rel.cst_pool_.at(0).cst_)
						&& dynamic_cast<const Motion*>(rel.cst_pool_.at(1).cst_)
						&& dynamic_cast<const Motion*>(rel.cst_pool_.at(1).cst_)->axis() == 2
						&& rel.cst_pool_.at(0).cst_->makI() == rel.cst_pool_.at(1).cst_->makI())
					{
						diag.upd_d_and_cp_ = Imp::prismatic_upd_d_and_cp;
						rel.dim_ = 6;
					}
					// Do not optimize //
					else{
						diag.upd_d_and_cp_ = Imp::normal_upd_d_and_cp;
					}
				}
			}

			// Produce r_vec (remainder_vec) //
			core::allocMem(mem_pool_size, sys.r_data_, rel_vec.size() - prt_vec.size() + 1);
			r_vec_vec.push_back(std::vector<LocalRemainder>());
			auto &r_vec = r_vec_vec.back();
			r_vec.clear();
			r_vec.resize(rel_vec.size() - prt_vec.size() + 1);
			for (Size i = 0; i < r_vec.size(); ++i){
				auto &r = r_vec.at(i);
				auto &rel = rel_vec.at(i + d_vec.size() - 1);

				r.cm_blk_series.clear();
				r.cm_blk_series.push_back(Remainder::Block());
				r.cm_blk_series.back().diag_ = &*std::find_if(d_vec.begin(), d_vec.end(), [&rel](Diag& d) {return rel.prtI_ == d.part_; });
				r.cm_blk_series.back().is_I_ = true;
				r.cm_blk_series.push_back(Remainder::Block());
				r.cm_blk_series.back().diag_ = &*std::find_if(d_vec.begin(), d_vec.end(), [&rel](Diag& d) {return rel.prtJ_ == d.part_; });
				r.cm_blk_series.back().is_I_ = false;

				for (auto rd = d_vec.rbegin(); rd < d_vec.rend() - 1; ++rd){
					auto &d = *rd;
					auto &d_rel = rel_vec.at(d_vec.rend() - rd - 2);

					auto diag_part = d_rel.prtI_;
					auto add_part = d_rel.prtJ_;

					// Determine if current remainder addition element exists (not 0)
					auto diag_blk = std::find_if(r.cm_blk_series.begin(), r.cm_blk_series.end(), [&](Remainder::Block &blk) {return blk.diag_->part_ == diag_part; });
					auto add_blk = std::find_if(r.cm_blk_series.begin(), r.cm_blk_series.end(), [&](Remainder::Block &blk) {return blk.diag_->part_ == add_part; });
					if (diag_blk != r.cm_blk_series.end()){
						if (add_blk != r.cm_blk_series.end()){
							r.cm_blk_series.erase(add_blk);
						}
						else{
							Remainder::Block blk;
							blk.is_I_ = diag_blk->is_I_;
							blk.diag_ = &*std::find_if(d_vec.begin(), d_vec.end(), [&](Diag &d) {
								return d.part_ == add_part; 
							});
							r.cm_blk_series.push_back(blk);
						}
					}
				}

				// Allocate Relation::Block memory
				core::allocMem(mem_pool_size, rel.blk_data_, rel.cst_pool_.size());

				// Allocate size of cmI_vec, cmJ_vec, bc_vec, xc_vec in Remainder
				core::allocMem(mem_pool_size, r_vec[i].cmI_, 6 * rel.size_);
				core::allocMem(mem_pool_size, r_vec[i].cmJ_, 6 * rel.size_);
				core::allocMem(mem_pool_size, r_vec[i].bc_, rel.size_);
				core::allocMem(mem_pool_size, r_vec[i].xc_, rel.size_);

				// Allocate Remainder::Block memory
				r_vec[i].blk_size_ = r.cm_blk_series.size();
				core::allocMem(mem_pool_size, r_vec[i].blk_data_, r.cm_blk_series.size());
			}
			
			// Update subsystem size //
			sys.fm_ = 0;
			sys.fn_ = 0;
			for (Size i = 1; i < d_vec.size(); ++i)sys.fm_ += rel_vec[i - 1].dim_;
			for (Size i = 0; i < r_vec.size(); ++i)sys.fn_ += rel_vec[i + d_vec.size() - 1].size_;

			sys.gm_ = sys.hasGround() ? sys.fm_ : sys.fm_ + 6;
			sys.gn_ = sys.hasGround() ? sys.fm_ : sys.fm_ + 6;

			max_F_size = std::max(max_F_size, sys.fm_ * sys.fn_);
			max_fm = std::max(max_fm, sys.fm_);
			max_fn = std::max(max_fn, sys.fn_);
			max_G_size = std::max(max_G_size, sys.gm_ * sys.gn_);
			max_gm = std::max(max_gm, sys.gm_);
			max_gn = std::max(max_gn, sys.gn_);
		}
		core::allocMem(mem_pool_size, pub_data.subsys_data_, sys_vec.size());

		// Calculate shared memory and offset required
		pub_data.active_mot_dim_ = active_mot_dim;
		pub_data.active_mot_size_ = active_mot_size;
		pub_data.active_mp_size_ = active_mp_size;
		pub_data.deactive_mot_dim_ = deactive_mot_dim;
		pub_data.deactive_mot_size_ = deactive_mot_size;
		pub_data.deactive_mp_size_ = deactive_mp_size;

		pub_data.mJg_ = model()->partPool().size() * 6;
		pub_data.nJg_ = active_mot_dim;
		pub_data.nM_ = active_mot_dim;

		core::allocMem(mem_pool_size, pub_data.active_mots_, pub_data.active_mot_size_);
		core::allocMem(mem_pool_size, pub_data.active_mp_, active_mp_size);
		core::allocMem(mem_pool_size, pub_data.deactive_mots_, pub_data.deactive_mot_size_);
		core::allocMem(mem_pool_size, pub_data.deactive_mp_, deactive_mp_size);
		core::allocMem(mem_pool_size, pub_data.cmI_, max_cm_size * 6);
		core::allocMem(mem_pool_size, pub_data.cmJ_, max_cm_size * 6);
		core::allocMem(mem_pool_size, pub_data.cmU_, max_cm_size * 6);
		core::allocMem(mem_pool_size, pub_data.cmT_, std::max(Size(6), max_cm_size));
		core::allocMem(mem_pool_size, pub_data.F_, max_F_size);
		core::allocMem(mem_pool_size, pub_data.FT_, std::max(max_fm, max_fn));
		core::allocMem(mem_pool_size, pub_data.FP_, std::max(max_fm, max_fn));
		core::allocMem(mem_pool_size, pub_data.G_, max_G_size);
		core::allocMem(mem_pool_size, pub_data.GT_, std::max(max_gm, max_gn));
		core::allocMem(mem_pool_size, pub_data.GP_, std::max(max_gm, max_gn));
		core::allocMem(mem_pool_size, pub_data.S_, max_fm * max_fm);
		core::allocMem(mem_pool_size, pub_data.beta_, max_gn);
		core::allocMem(mem_pool_size, pub_data.xcf_, std::max(max_fn, max_fm));
		core::allocMem(mem_pool_size, pub_data.xpf_, std::max(max_fn, max_fm));
		core::allocMem(mem_pool_size, pub_data.bcf_, max_fn);
		core::allocMem(mem_pool_size, pub_data.bpf_, max_fm);
		core::allocMem(mem_pool_size, pub_data.Jg_, pub_data.mJg_* pub_data.nJg_);
		core::allocMem(mem_pool_size, pub_data.cg_, pub_data.mJg_);
		core::allocMem(mem_pool_size, pub_data.M_, pub_data.nM_ * pub_data.nM_);
		core::allocMem(mem_pool_size, pub_data.h_, pub_data.nM_);
		core::allocMem(mem_pool_size, pub_data.get_diag_from_part_id_, model()->partPool().size());

		// Allocate memory
		imp_->mem_pool_.resize(mem_pool_size);

		// Update public variable block //
		{
			imp_->pd_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_);
			*imp_->pd_ = pub_data;

			imp_->pd_->active_mots_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->active_mots_);
			imp_->pd_->deactive_mots_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->deactive_mots_);
			imp_->pd_->active_mp_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->active_mp_);
			imp_->pd_->deactive_mp_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->deactive_mp_);

			// Obtain memory for the Jacobian part //
			imp_->pd_->Jg_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->Jg_);
			imp_->pd_->cg_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->cg_);
			imp_->pd_->M_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->M_);
			imp_->pd_->h_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->h_);

			imp_->pd_->F_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->F_);
			imp_->pd_->FU_ = imp_->pd_->F_;
			imp_->pd_->FT_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->FT_);
			imp_->pd_->FP_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->FP_);
			imp_->pd_->G_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->G_);
			imp_->pd_->GU_ = imp_->pd_->G_;
			imp_->pd_->GT_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->GT_);
			imp_->pd_->GP_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->GP_);
			imp_->pd_->QT_DOT_G_ = imp_->pd_->G_;
			imp_->pd_->S_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->S_);
			imp_->pd_->beta_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->beta_);
			imp_->pd_->xcf_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->xcf_);
			imp_->pd_->xpf_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->xpf_);
			imp_->pd_->bcf_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->bcf_);
			imp_->pd_->bpf_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->bpf_);
			imp_->pd_->cmI_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->cmI_);
			imp_->pd_->cmJ_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->cmJ_);
			imp_->pd_->cmU_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->cmU_);
			imp_->pd_->cmT_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->cmT_);
		}

		// mots //
		std::copy_n(active_mot_vec.data(), active_mot_vec.size(), imp_->pd_->active_mots_);
		std::copy_n(deactive_mot_vec.data(), deactive_mot_vec.size(), imp_->pd_->deactive_mots_);

		// Assign memory to subsystem and initialize //
		for (int i = 0; i < sys_vec.size(); ++i){
			auto &sys = sys_vec[i];
			auto &prt_vec = prt_vec_vec[i];
			auto &rel_vec = rel_vec_vec[i];
			auto &d_vec = d_vec_vec[i];
			auto &r_vec = r_vec_vec[i];

			sys.pd_ = imp_->pd_;

			// Update diags //
			sys.d_data_ = core::getMem(imp_->mem_pool_.data(), sys.d_data_);
			sys.d_size_ = d_vec.size();
			for (int i = 0; i < sys.d_size_; ++i) {
				sys.d_data_[i] = d_vec[i];
				sys.d_data_[i].cmI_ = imp_->pd_->cmI_;
				sys.d_data_[i].cmJ_ = imp_->pd_->cmJ_;
				sys.d_data_[i].cmU_ = imp_->pd_->cmU_;
				sys.d_data_[i].cmT_ = imp_->pd_->cmT_;
			}
			sys.d_data_[0].pm_ = sys.d_data_[0].pm1_;
			sys.d_data_[0].last_pm_ = sys.d_data_[0].pm2_;
			for (Size i = 1; i < sys.d_size_; ++i){
				auto &diag = sys.d_data_[i];
				auto &rel = rel_vec.at(i - 1);

				// Obtain Block::Relation memory //
				{
					rel.blk_data_ = core::getMem(imp_->mem_pool_.data(), rel.blk_data_);
					rel.blk_size_ = rel.cst_pool_.size();
					std::copy(rel.cst_pool_.data(), rel.cst_pool_.data() + rel.cst_pool_.size(), rel.blk_data_);
					diag.rel_ = static_cast<aris::dynamic::Relation>(rel);
				}
				
				// Obtain p_vec memory //
				diag.p_ = core::getMem(imp_->mem_pool_.data(), diag.p_);
				diag.bc_ = core::getMem(imp_->mem_pool_.data(), diag.bc_);
				diag.xc_ = core::getMem(imp_->mem_pool_.data(), diag.xc_);
				std::iota(diag.p_, diag.p_ + rel.size_, 0);

				// Update mp position corresponding to blk
				ARIS_LOOP_BLOCK(diag.rel_.) {
					b->mp_ = imp_->pd_->active_mp_ + b->mot_mp_pos_;
				}

				// Initialize diag //
				diag.rd_ = std::find_if(sys.d_data_, sys.d_data_ + sys.d_size_, [&](Diag &d) {return d.part_ == rel.prtJ_; });
				diag.pm_ = diag.pm1_;
				diag.last_pm_ = diag.pm2_;
			}

			// Update remainders //
			sys.r_data_ = reinterpret_cast<Remainder*>(imp_->mem_pool_.data() + *reinterpret_cast<Size*>(&sys.r_data_));
			sys.r_size_ = r_vec.size();
			for(int i = 0; i < sys.r_size_; ++i)sys.r_data_[i] = static_cast<aris::dynamic::Remainder>(r_vec[i]);
			for (Size i = 0; i < sys.r_size_; ++i){
				auto &r = sys.r_data_[i];
				auto &rel = rel_vec[i + sys.d_size_ - 1];

				// Obtain Relation::Block memory
				{
					rel.blk_data_ = core::getMem(imp_->mem_pool_.data(), rel.blk_data_);
					rel.blk_size_ = rel.cst_pool_.size();
					std::copy(rel.cst_pool_.data(), rel.cst_pool_.data() + rel.cst_pool_.size(), rel.blk_data_);
					r.rel_ = static_cast<aris::dynamic::Relation>(rel);
				}

				// Allocate size of cmI_, cmJ_, bc_, xc_ in Remainder
				r.cmI_ = core::getMem(imp_->mem_pool_.data(), r.cmI_);
				r.cmJ_ = core::getMem(imp_->mem_pool_.data(), r.cmJ_);
				r.bc_ = core::getMem(imp_->mem_pool_.data(), r.bc_);
				r.xc_ = core::getMem(imp_->mem_pool_.data(), r.xc_);

				// Obtain Remainder::Block memory
				r.blk_data_ = core::getMem(imp_->mem_pool_.data(), r.blk_data_);
				for (int j = 0; j < r.blk_size_; ++j){
					r.blk_data_[j] = r_vec[i].cm_blk_series[j];
					r.blk_data_[j].diag_ = sys.d_data_ + (r.blk_data_[j].diag_ - d_vec.data());
				}

				// Update mp //
				ARIS_LOOP_BLOCK(r.rel_.) {
					b->mp_ = imp_->pd_->active_mp_ + b->mot_mp_pos_;
				}
				
				// Construct r
				r.i_diag_ = std::find_if(sys.d_data_, sys.d_data_ + sys.d_size_, [&rel](Diag& d) {return rel.prtI_ == d.part_; });
				r.j_diag_ = std::find_if(sys.d_data_, sys.d_data_ + sys.d_size_, [&rel](Diag& d) {return rel.prtJ_ == d.part_; });
			}
		}

		// Allocate vector to find diag based on part id //
		imp_->pd_->get_diag_from_part_id_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->get_diag_from_part_id_);
		for (auto &sys : sys_vec)
			for (auto diag = sys.d_data_; diag < sys.d_data_ + sys.d_size_; ++diag)
				imp_->pd_->get_diag_from_part_id_[diag->part_->id()] = diag;

		imp_->pd_->subsys_size_ = sys_vec.size();
		imp_->pd_->subsys_data_ = core::getMem(imp_->mem_pool_.data(), imp_->pd_->subsys_data_);
		std::copy_n(sys_vec.data(), sys_vec.size(), imp_->pd_->subsys_data_);
	}
	auto UniversalSolver::kinPos()->int{
		kinPosSetMotionPosFromModel();
		if (auto ret = kinPosCompute())
		  // error not converge with iteration
			return ret;
		else {
			kinPosUpdateModel();
			return ret;
		}
	}
	auto UniversalSolver::kinVel()->int	{
		ARIS_LOOP_SYS ARIS_LOOP_SYS_D d->part_->getPm(d->pm_);

		s_fill(6, 1, 0.0, const_cast<double *>(model()->ground().vs()));
		ARIS_LOOP_SYS sys->kinVel();

		// Calculation successful, set parts //
		ARIS_LOOP_SYS ARIS_LOOP_SYS_D s_va(6, d->xp_, const_cast<double*>(d->part_->vs()));

		return 0;
	}
	auto UniversalSolver::dynAccAndFce()->int{
		// Update part pose, external force of each part //
		ARIS_LOOP_SYS ARIS_LOOP_SYS_D {
			d->part_->getPm(d->pm_);
			std::fill(d->bp_, d->bp_ + 6, 0.0);
		}

		// Update external force //
		for (auto &fce : model()->forcePool()){
			if (fce.active()){
				double fsI[6], fsJ[6];
				fce.cptGlbFs(fsI, fsJ);

				if (&fce.makI()->fatherPart() != &model()->ground()) 
					s_vs(6, fsI, imp_->pd_->get_diag_from_part_id_[fce.makI()->fatherPart().id()]->bp_);
				
				if (&fce.makJ()->fatherPart() != &model()->ground())
					s_vs(6, fsJ, imp_->pd_->get_diag_from_part_id_[fce.makJ()->fatherPart().id()]->bp_);
			}
		}

		// Update ground as //
		s_fill(6, 1, 0.0, const_cast<double *>(model()->ground().as()));
		ARIS_LOOP_SYS sys->dynAccAndFce();

		// Calculation successful, set joints and parts
		ARIS_LOOP_SYS {
			ARIS_LOOP_SYS_R	{
				Size pos{ 0 };
				ARIS_LOOP_BLOCK(r->rel_.) {
					const_cast<Constraint*>(b->cst_)->setCf(r->xc_ + pos);
					pos += b->cst_->dim();
				}
			}
			for (auto d = sys->d_data_ + 1; d<sys->d_data_ + sys->d_size_; ++d) {
				Size pos{ 0 };
				ARIS_LOOP_BLOCK(d->rel_.) {
					const_cast<Constraint*>(b->cst_)->setCf(d->xc_ + pos);
					pos += b->cst_->dim();
				}
			}

			ARIS_LOOP_SYS_D s_vc(6, d->xp_, const_cast<double*>(d->part_->as()));
		}

		return 0;
	}
	auto UniversalSolver::kinPosPure(const double* motion_pos, double* answer, int which_root, const double* current_answer)->int {
		kinPosSetActiveMotionPos(motion_pos);
		if (auto ret = kinPosCompute())
			return ret;
		else {
			kinPosGetUnactiveMotionPos(answer);
			return ret;
		}
	}
	auto UniversalSolver::whichRootOfAnswer(const double* motion_pos, const double* answer)->int {
		int solution_id = 0;
		double error = std::numeric_limits<double>::infinity();

		if (rootNumber() == 1) {
			return 0;
		}
		else {
			for (int i = 0; i < rootNumber(); ++i) {
				kinPosPure(motion_pos, imp_->pd_->deactive_mp_, i, answer);
				aris::Size pos = 0;
				double this_error = 0.0;
				for (int j = 0; j < imp_->pd_->deactive_mot_size_; ++j) {
					this_error = std::max(this_error, imp_->pd_->deactive_mots_[j]->cptPError(imp_->pd_->deactive_mp_ + pos, answer + pos));
					pos += imp_->pd_->deactive_mots_[j]->pSize();
				}

				if (this_error < error) {
					error = this_error;
					solution_id = i;
				}
			}

			return solution_id;
		}
	}
	auto UniversalSolver::answerSize()->aris::Size {
		return imp_->pd_->deactive_mp_size_;
	}
	auto UniversalSolver::kinPosGetUnactiveMotionPos(double* mp)->void {
		for (Size i = 0, mp_pos = 0; i < imp_->pd_->deactive_mot_size_; i++) {
			auto mot = imp_->pd_->deactive_mots_[i];
			auto blk_i = imp_->pd_->get_diag_from_part_id_[mot->makI()->fatherPart().id()];
			auto blk_j = imp_->pd_->get_diag_from_part_id_[mot->makJ()->fatherPart().id()];
			
			double mak_pm_i[16], mak_pm_j[16];
			s_pm_dot_pm(blk_i->pm_, *mot->makI()->prtPm(), mak_pm_i);
			s_pm_dot_pm(blk_j->pm_, *mot->makJ()->prtPm(), mak_pm_j);
			
			double pm_i2j[16];
			s_inv_pm_dot_pm(mak_pm_j, mak_pm_i, pm_i2j);

			mot->cptPFromPm(pm_i2j,	mp + mp_pos);
			mp_pos += mot->pSize();
		}
	}
	auto UniversalSolver::kinPosSetActiveMotionPos(const double* mp)->void {
		// Copy the positions of drives and part poses into local variables //
		s_vc(imp_->pd_->active_mp_size_, mp, imp_->pd_->active_mp_);
	}
	auto UniversalSolver::kinPosSetMotionPosFromModel()->void {
		for (Size i = 0, mp_pos = 0; i< imp_->pd_->active_mot_size_; i++) {
			imp_->pd_->active_mots_[i]->getP(imp_->pd_->active_mp_ + mp_pos);
			mp_pos += imp_->pd_->active_mots_[i]->pSize();
		}
	}
	auto UniversalSolver::kinPosCompute()->int {
		const double pm[16]{ 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1 };
		s_mc(4, 4, pm, const_cast<double*>(*model()->ground().pm()));

		// Copy part poses into local variables //
		ARIS_LOOP_SYS ARIS_LOOP_SYS_D d->part_->getPm(d->pm_);

		setError(0.0);
		setIterCount(0);

		ARIS_LOOP_SYS{
			sys->max_error_ = maxError();
			sys->max_iter_count_ = maxIterCount();
			sys->kinPos();

			setIterCount(std::max(iterCount(), sys->iter_count_));
			setError(std::max(error(), sys->error_));
		}

		return error() < maxError() ? 0 : -1;
	}
	auto UniversalSolver::kinPosUpdateModel()->void {
		ARIS_LOOP_SYS ARIS_LOOP_SYS_D const_cast<Part*>(d->part_)->setPm(d->pm_);
	}
	auto UniversalSolver::cptGeneralJacobi()noexcept->void{
		auto Jg = imp_->pd_->Jg_;
		auto cg = imp_->pd_->cg_;
		
		std::fill(Jg, Jg + mJg() * nJg(), 0.0);
		std::fill(cg, cg + mJg(), 0.0);

		ARIS_LOOP_SYS {
			ARIS_LOOP_SYS_D d->part_->getPm(d->pm_);

			// make A
			sys->updDmCm(false);

			// solve
			sys->updF();
			ARIS_LOOP_SYS_D std::fill_n(d->bc_, d->rel_.size_, 0.0);
			ARIS_LOOP_SYS_R std::fill_n(r->bc_, r->rel_.size_, 0.0);

			// upd Jg //
			auto getJacobiColumn = [&](Relation &rel, double *bc){
				Size pos = 0;
				ARIS_LOOP_BLOCK(rel.){
					if (auto gmt = dynamic_cast<const MotionBase*>(b->cst_)) {
						// reserve old value
						double mv_store[6];
						gmt->getV(mv_store);

						for (Size k(-1); ++k < gmt->dim();) {
							double mv[11]{ 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
							const_cast<MotionBase*>(gmt)->setV(mv + 5 - k);
							double cv[6];
							gmt->cptCv(cv);

							s_vc(gmt->dim(), cv, bc + pos);
							sys->sovXp();
							ARIS_LOOP_SYS_D s_vc(6, d->xp_, 1, Jg + at(d->part_->id() * 6, b->mot_dim_pos_ + k, nJg()), nJg());
						}

						// restore to old value
						const_cast<MotionBase*>(gmt)->setV(mv_store);
						std::fill_n(bc + pos, gmt->dim(), 0.0);
					}

					pos += b->cst_->dim();
				}
			};
			ARIS_LOOP_SYS_D getJacobiColumn(d->rel_, d->bc_);
			ARIS_LOOP_SYS_R getJacobiColumn(r->rel_, r->bc_);

			// upd cg //
			sys->updCa();
			auto clearMotionMa = [&](Relation &rel, double *bc){
				Size pos = 0;
				ARIS_LOOP_BLOCK(rel.) {
					if (auto mot = dynamic_cast<const MotionBase*>(b->cst_)) {
						double ca0[6], ca[6];
						b->cst_->cptCa(ca);

						double ma_store[6];
						mot->getA(ma_store);
						double ma[6]{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
						const_cast<MotionBase*>(mot)->setA(ma);

						b->cst_->cptCa(ca0);

						s_vs(b->cst_->dim(), ca, bc + pos);
						s_va(b->cst_->dim(), ca0, bc + pos);

						const_cast<MotionBase*>(mot)->setA(ma_store);
					}
					
					pos += b->cst_->dim();
				}
			};
			ARIS_LOOP_SYS_D clearMotionMa(d->rel_, d->bc_);
			ARIS_LOOP_SYS_R clearMotionMa(r->rel_, r->bc_);
			sys->sovXp();
			ARIS_LOOP_SYS_D s_vc(6, d->xp_, cg + d->part_->id() * 6);
		}
	}
	auto UniversalSolver::mJg()const noexcept->Size { return model()->partPool().size() * 6; }
	auto UniversalSolver::nJg()const noexcept->Size { return imp_->pd_->nJg_; }
	auto UniversalSolver::Jg()const noexcept->const double * { return imp_->pd_->Jg_; }
	auto UniversalSolver::cg()const noexcept->const double * { return imp_->pd_->cg_; }
	auto UniversalSolver::cptGeneralInverseDynamicMatrix()noexcept->void{
		auto M = imp_->pd_->M_;
		auto h = imp_->pd_->h_;
		
		// init //
		std::fill(M, M + nM()* nM(), 0.0);
		std::fill(h, h + nM(), 0.0);

		ARIS_LOOP_SYS{
			ARIS_LOOP_SYS_D d->part_->getPm(d->pm_);
			
			// Kinematics/Dynamics calculation, exactly the same as dynAccAndFce() //
			sys->updDiagIv();
			sys->updDmCm(false);

			sys->updF();
			sys->updG();
			sys->updCa();

			auto dynamic = [this, sys](){
				ARIS_LOOP_SYS ARIS_LOOP_SYS_D std::fill(d->bp_, d->bp_ + 6, 0.0);
				for (auto &fce : this->model()->forcePool()){
					if (fce.active()){
						double fsI[6], fsJ[6];
						fce.cptGlbFs(fsI, fsJ);
						if (&fce.makI()->fatherPart() != &model()->ground())
							s_vs(6, fsI, imp_->pd_->get_diag_from_part_id_[fce.makI()->fatherPart().id()]->bp_);
						
						if (&fce.makJ()->fatherPart() != &model()->ground())
							s_vs(6, fsJ, imp_->pd_->get_diag_from_part_id_[fce.makJ()->fatherPart().id()]->bp_);
					}
				}
				
				sys->sovXp();
				sys->sovXc();
			};

			// Start calculating h //
			// First remove the acceleration of drive, and calculate h
			auto clearMotionMa = [](Relation &rel, double *bc){
				Size pos = 0;
				ARIS_LOOP_BLOCK(rel.){
					if (auto gm = dynamic_cast<const MotionBase*>(b->cst_)){
						// reserve old value //
						double ma_old[6];
						s_vc(gm->dim(), gm->a(), ma_old);

						// Calculate ca caused by drive //
						double old_ca[6];
						gm->cptCa(old_ca);

						const_cast<MotionBase*>(gm)->setA(std::array<double, 6>{0, 0, 0, 0, 0, 0}.data());
						double ca[6];
						gm->cptCa(ca);

						s_vs(gm->dim(), ca, old_ca);

						// Add to bc //
						s_vs(gm->dim(), old_ca, bc + pos);

						// restore to old value //
						const_cast<MotionBase*>(gm)->setA(ma_old);
					}

					pos += b->cst_->dim();
				}
			};
			ARIS_LOOP_SYS_D clearMotionMa(d->rel_, d->bc_);
			ARIS_LOOP_SYS_R clearMotionMa(r->rel_, r->bc_);

			// Kinematics calculation and extract h
			dynamic();
			auto getH = [&](Relation &rel, double *xc){
				Size pos{ 0 };
				// Update Xcf //
				ARIS_LOOP_BLOCK(rel.){
					if (auto gm = dynamic_cast<const MotionBase*>(b->cst_)){
						s_vc(gm->dim(), xc + pos, h + b->mot_dim_pos_);
					}
					pos += b->cst_->dim();
				}
			};
			ARIS_LOOP_SYS_D getH(d->rel_, d->xc_);
			ARIS_LOOP_SYS_R getH(r->rel_, r->xc_);

			// Start calculating M //
			auto getMColumn = [&](const Constraint *c, Size cid){
				auto nM = this->nM();
				auto getMRow = [&](Relation &rel, double *xc){
					Size pos2{ 0 };
					ARIS_LOOP_BLOCK(rel.){
						if (auto gm = dynamic_cast<const MotionBase*>(b->cst_)){
							Size ccid = b->mot_dim_pos_;
							s_vc(gm->dim(), xc + pos2, 1, M + at(ccid, cid, nM), nM);
							s_vs(gm->dim(), h + ccid, 1, M + at(ccid, cid, nM), nM);
						}
						pos2 += b->cst_->dim();
					}
				};
				ARIS_LOOP_SYS_R getMRow(r->rel_, r->xc_);
				ARIS_LOOP_SYS_D getMRow(d->rel_, d->xc_);
			};
			auto getM = [&](Relation &rel, double *bc){
				Size pos{ 0 };
				ARIS_LOOP_BLOCK(rel.){
					if (auto gm = dynamic_cast<const MotionBase*>(b->cst_)){
						// reserve old value //
						double old_ma[6];
						s_vc(gm->dim(), gm->a(), old_ma);
						
						for (Size i = 0; i < gm->dim(); ++i){
							// old ca, all 0 //
							const_cast<MotionBase*>(gm)->setA(std::array<double, 6>{0, 0, 0, 0, 0, 0}.data());
							double caa[6];
							gm->cptCa(caa);
							
							// new ca, unit //
							const double ma_unit[11]{ 0,0,0,0,0,1,0,0,0,0,0 };
							const_cast<MotionBase*>(gm)->setA(ma_unit + 5 - i);
							double new_ca[6];
							gm->cptCa(new_ca);

							// diff between old & new //
							s_vs(gm->dim(), caa, new_ca);

							// Add to bc //
							s_va(gm->dim(), new_ca, bc + pos);

							dynamic();
							getMColumn(b->cst_, b->mot_dim_pos_ + i);

							s_vs(gm->dim(), new_ca, bc + pos);
						}

						// restore to old value //
						const_cast<MotionBase*>(gm)->setA(old_ma);
					}
					pos += b->cst_->dim();
				}
			};
			ARIS_LOOP_SYS_D getM(d->rel_, d->bc_);
			ARIS_LOOP_SYS_R getM(r->rel_, r->bc_);
		}
	}
	auto UniversalSolver::nM()const noexcept->Size { return imp_->pd_->nM_; }
	auto UniversalSolver::M()const noexcept->const double * { return imp_->pd_->M_; }
	auto UniversalSolver::h()const noexcept->const double * { return imp_->pd_->h_; }
	auto UniversalSolver::cptProjectedMassMatrix() noexcept -> void {
		// Method 1, direct construction
// 		auto M = imp_->pd_->M_;
// 		auto h = imp_->pd_->h_;
		
// 		// init //
// 		std::fill(M, M + nM()* nM(), 0.0);
// 		std::fill(h, h + nM(), 0.0);

// 		ARIS_LOOP_SYS{
// 			ARIS_LOOP_SYS_D d->part_->getPm(d->pm_);
			
// 			// Kinematics/Dynamics calculation, exactly the same as dynAccAndFce() //
// 			sys->updDiagIv();
// 			sys->updDmCm(false);

// 			sys->updF();
// 			sys->updG();
// 			sys->updCa();
// 		// Allocate K and M_beta_K at the beginning of updG()
// Eigen::MatrixXd M_beta_K(d, d);
// std::vector<std::vector<Eigen::Vector6d>> K_cols(d, std::vector<Eigen::Vector6d>(d_size_));

// for (Size j = 0; j < d; ++j) {
//     // ... Existing code: set xpf, transform to xp (but do not multiply by I first)...
//     // Save the j-th column of K
//     for (Size i = 0; i < d_size_; ++i) {
//         K_cols[j][i] = Eigen::Vector6d(d_data_[i].xp_);
//     }
//     // Then multiply by I and continue building G...
// }

// // Assemble M_beta_K
// for (Size i = 0; i < d; ++i) {
//     for (Size j = 0; j <= i; ++j) {
//         double val = 0.0;
//         for (Size part = 0; part < d_size_; ++part) {
//             // Calculate f_j_part = I_part * K_cols[j][part]
//             Eigen::Vector6d f_j_part;
//             s_iv_dot_as(d_data_[part].iv_, K_cols[j][part].data(), f_j_part.data());
//             val += K_cols[i][part].dot(f_j_part);
//         }
//         M_beta_K(i, j) = val;
//         M_beta_K(j, i) = val;
//     }
// }
		// Method 2, direct derivation via matrix G
		ARIS_LOOP_SYS{
			ARIS_LOOP_SYS_D d->part_->getPm(d->pm_);
			
			// Kinematics/Dynamics calculation, exactly the same as dynAccAndFce() //
			sys->updDiagIv();
			sys->updDmCm(false);

			sys->updF();
			sys->updG();
			sys->updCa();
			sys->sovXp();
			sys->sovProjectMassMatrix();
			aris::dynamic::dsp(sys->gn_, sys->gn_, sys->pd_->QT_DOT_G_);
		}
	}
	auto UniversalSolver::cptContactInverseInertiaMatrix(int nContact, int* partid, double* T_vec,
		double* contactPoint, std::vector<double>& A_out, std::vector<double>& accel0) noexcept -> void {
		auto* pd = imp_->pd_;
    int num_subsys = pd->subsys_size_;
    SubSystem* subsys = pd->subsys_data_;
		int ground_id = model()->ground().id();

		auto inverse_pd = [](Size m, double* A) {
		  // Assume A and invA are row-major, leading dimension is m
		  std::vector<double> L(m * m);
		  s_llt(m, A, m, L.data(), m);  // A = L * L^T
		  std::vector<double> invL(m * m);
		  s_inv_lm(m, L.data(), m, invL.data(), m);  // invL = L^{-1}
		  // Calculate invA = invL^T * invL
		  // Use s_mm: C = alpha * A * B, where alpha=1, A = invL^T (column-major?), B = invL (row-major)
		  // s_mm Function signature: s_mm(m, n, k, alpha, A, a_t, B, b_t, C, c_t)
		  // We calculate m x m matrix invL^T * invL:
		  //  A is transpose of invL, interpreted as column-major, i.e., a_t = T(m) (since after transpose leading dim is m, and in aris T(ColMajor(m)) or T(m) corresponds to column-major)
		  // Or we could loop directly or use s_mm providing appropriate types.
		  // Convenient way: notice invA is symmetric, can use s_mm(m, m, m, invL, m, invL, T(m), invA, m)? Need to check.
		  // In aris, invL is stored row-major, i.e., RowMajor(m). Its transpose is equivalent to ColMajor(m). So in s_mm, we can specify a_t = T(m), b_t = m, c_t = m.
		  s_mm(m, m, m, 1.0, invL.data(), T(m), invL.data(), m, A, m);
		};
		// 2. Determine the subsystems and local body indices for the two objects in each contact
    struct ContactInfo {
      int sys1, sys2;          // Subsystem index, -1 means the object is ground
      int local_body1, local_body2; // Index in subsystem d_data_, ground is meaningless
			double* T;
      double* pt;
    };
    std::vector<ContactInfo> contact_info(nContact);
   	for (int ic = 0; ic < nContact; ++ic) {
			int p1 = partid[2 * ic], p2 = partid[2 * ic + 1];
      // Look up affiliated subsystem
      int s1 = -1, s2 = -1, lb1 = -1, lb2 = -1;
			if (p1 == ground_id) 
				s1 = -1;
			else {
      	Diag* d1 = pd->get_diag_from_part_id_[partid[2 * ic]];
      	ARIS_LOOP_SYS {
      	  if (d1 >= sys->d_data_ && d1 < sys->d_data_ + sys->d_size_) {
      	    s1 = sys - subsys; lb1 = d1 - subsys->d_data_; break;
      	  }
      	}
			}
			if (p2 == ground_id)
				s2 = -1;
			else {
      	Diag* d2 = pd->get_diag_from_part_id_[partid[2 * ic + 1]];
      	ARIS_LOOP_SYS {
          if (d2 >= sys->d_data_ && d2 < sys->d_data_ + sys->d_size_) {
            s2 = sys - subsys; lb2 = d2 - subsys->d_data_; break;
          }
      	}
			}
			contact_info[ic] = {s1, s2, lb1, lb2, {T_vec + ic * 16}, {contactPoint + 3*ic}};
    }
    // 3. Allocate output matrix
    const int rows_per_contact = 6;   // Object 1: 3 rows, object 2: 3 rows
    const int J_rows = rows_per_contact * nContact;
    A_out.resize(J_rows * J_rows, 0.0);

		// Collect non-ground object tasks involved in this subsystem
    struct Task {
      int contact_idx;   // Global contact index
      int obj_idx;       // 0 indicates Object 1, 1 indicates Object 2
      int local_body;    // Body index within the subsystem
			double* T;
      double* point;
    };
		ARIS_LOOP_SYS ARIS_LOOP_SYS_D {
			d->part_->getPm(d->pm_);
			std::fill(d->bp_, d->bp_ + 6, 0.0);
		}
		// Update external force //
		for (auto &fce : model()->forcePool()){
			if (fce.active()){
				double fsI[6], fsJ[6];
				fce.cptGlbFs(fsI, fsJ);

				if (&fce.makI()->fatherPart() != &model()->ground()) 
					s_vs(6, fsI, imp_->pd_->get_diag_from_part_id_[fce.makI()->fatherPart().id()]->bp_);
				
				if (&fce.makJ()->fatherPart() != &model()->ground())
					s_vs(6, fsJ, imp_->pd_->get_diag_from_part_id_[fce.makJ()->fatherPart().id()]->bp_);
			}
		}
		// Update ground as //
		s_fill(6, 1, 0.0, const_cast<double *>(model()->ground().as()));

		ARIS_LOOP_SYS{
			std::vector<Task> tasks;
    	std::set<int> involvedLocalBodies;
			Size s = sys - subsys;
    	for (int ic = 0; ic < nContact; ++ic) {
    	  const auto& info = contact_info[ic];
    	  if (info.sys1 == s) {  // Only non-ground (sys1 != -1)
    	    tasks.push_back({ic, 0, info.local_body1, info.T, info.pt});
    	    involvedLocalBodies.insert(info.local_body1);
    	  }
    	  if (info.sys2 == s) {
    	    tasks.push_back({ic, 1, info.local_body2, info.T, info.pt});
    	    involvedLocalBodies.insert(info.local_body2);
    	  }
    	}
			if (tasks.empty()) continue;  // This subsystem has no related non-ground objects

			// ARIS_LOOP_SYS_D {
			// 	d->part_->getPm(d->pm_); 
			// 	// std::fill(d->bp_, d->bp_ + 6, 0.0);
			// }
			
			// Kinematics/Dynamics calculation, exactly the same as dynAccAndFce() //
			sys->updDiagIv();
			sys->updDmCm(false);

			sys->updF();
			sys->updG();
			sys->updCa();
			sys->sovXp();
			sys->sovProjectMassMatrix();
			Size gn = sys->gn_;
			std::vector<double> invM_beta(gn * gn);
			aris::dynamic::s_vc(gn * gn, sys->pd_->QT_DOT_G_, invM_beta.data());
			// aris::dynamic::dsp(gn, gn, invM_beta.data());
			inverse_pd(gn, invM_beta.data());
			// aris::dynamic::dsp(gn, gn, invM_beta.data());

			// ---- 4.3 Numerical perturbation of beta, extract point Jacobian for desired objects ----
      std::vector<Size> targetBodies(involvedLocalBodies.begin(), involvedLocalBodies.end());
      std::unordered_map<int,int> bodyToPos;
      for (size_t t = 0; t < targetBodies.size(); ++t)
        bodyToPos[targetBodies[t]] = t;

       // Save/clear velocity
      std::vector<std::array<double,6>> save_xp(sys->d_size_);
      for (int b = 0; b < sys->d_size_; ++b)
        std::copy_n(sys->d_data_[b].xp_, 6, save_xp[b].begin());

			// ---- 3.3 Construct Jacobian segment J_sub (J_rows x n_beta) for this subsystem ----
      std::vector<double> J_sub(J_rows * gn, 0.0);
      std::vector<double> xpf(sys->fm_, 0.0);
      std::vector<double> target_xp(targetBodies.size() * 6);

      // Perturb beta
      for (int j = 0; j < sys->gn_; ++j) {
        for (int b = 0; b < sys->d_size_; ++b)
          std::fill_n(sys->d_data_[b].xp_, 6, 0.0);

        if (j < sys->fm_ - sys->fr_) {
          std::vector<double> beta_pert(sys->fm_ - sys->fr_, 0.0);
          beta_pert[j] = 1.0;
          s_mm(sys->fm_, 1, sys->fm_ - sys->fr_, sys->pd_->S_, beta_pert.data(), xpf.data());
          for (auto d = sys->d_data_ + 1; d < sys->d_data_ + sys->d_size_; ++d) {
            s_mma(6, 1, 6 - d->rel_.dim_,
                  d->dm_ + at(0, d->rel_.dim_, T(6)), T(6),
                  xpf.data() + d->rows_, 1, d->xp_, 1);
            s_va(6, d->rd_->xp_, d->xp_);
          }
        } else {
          int base_idx = j - (sys->fm_ - sys->fr_);
          sys->d_data_[0].xp_[base_idx] = 1.0;
          for (auto d = sys->d_data_ + 1; d < sys->d_data_ + sys->d_size_; ++d) {
            s_va(6, d->rd_->xp_, d->xp_);
          }
        }

        // Extract target body velocity
        for (size_t t = 0; t < targetBodies.size(); ++t) {
          int b = targetBodies[t];
          std::copy_n(sys->d_data_[b].xp_, 6, target_xp.data() + 6*t);
        }

        // Fill into the j-th column of J_sub
        for (const auto& task : tasks) {
          int pos = bodyToPos[task.local_body];
          const double* v = target_xp.data() + 6*pos;
					double vp_o[3], vp_c[3];
					s_vs2vp(v, task.point, vp_o);
          // double wx = v[3], wy = v[4], wz = v[5];
          // double cross_x = wy * task.point[2] - wz * task.point[1];
          // double cross_y = wz * task.point[0] - wx * task.point[2];
          // double cross_z = wx * task.point[1] - wy * task.point[0];
          // double vel_x = v[0] + cross_x;
          // double vel_y = v[1] + cross_y;
          // double vel_z = v[2] + cross_z;

					s_inv_pm_dot_v3(task.T, vp_o, vp_c);
          int row0 = task.contact_idx * 6 + task.obj_idx * 3;
          J_sub[(row0 + 0) * gn + j] = vp_c[0];
          J_sub[(row0 + 1) * gn + j] = vp_c[1];
          J_sub[(row0 + 2) * gn + j] = vp_c[2];
        }
      }

      // Restore velocity
      for (int b = 0; b < sys->d_size_; ++b)
        std::copy_n(save_xp[b].begin(), 6, sys->d_data_[b].xp_);

      // ---- 3.4 Accumulate A += J_sub * M_inv * J_sub^T ----
      for (int i = 0; i < J_rows; ++i) {
        for (int k = 0; k < gn; ++k) {
          double tmp = 0.0;
          for (int l = 0; l < gn; ++l)
            tmp += J_sub[i * gn + l] * invM_beta[l * gn + k];
          for (int j = i; j < J_rows; ++j) 
            A_out[i * J_rows + j] += tmp * J_sub[j * gn + k];
        }
      }
			sys->sovXcRemain();
    }
		
    // Symmetrically fill lower triangle
    for (int i = 0; i < J_rows; ++i)
      for (int j = i+1; j < J_rows; ++j)
        A_out[j * J_rows + i] = A_out[i * J_rows + j];

		// ARIS_LOOP_SYS sys->dynAccAndFce();

		// Calculation successful, set joints and parts
		ARIS_LOOP_SYS 
			ARIS_LOOP_SYS_D 
				s_vc(6, d->xp_, const_cast<double*>(d->part_->as()));
		
		// dynAccAndFce();
		accel0.resize(J_rows, 0.0);
		for (int ic = 0; ic < nContact; ++ic) {
			if (contact_info[ic].sys1 != -1) {
				// auto& d = subsys[contact_info[ic].sys1].d_data_[contact_info[ic].local_body1];
				auto& part = model()->partPool()[partid[2 * ic]];
				double ap_o[3];
				s_as2ap(part.vs(), part.as(), contact_info[ic].pt, ap_o);
				s_inv_pm_dot_v3(contact_info[ic].T, ap_o, accel0.data() + ic * rows_per_contact);
			}
			if (contact_info[ic].sys2 != -1) {
				// auto& d = subsys[contact_info[ic].sys2].d_data_[contact_info[ic].local_body2];
				auto& part = model()->partPool()[partid[2 * ic + 1]];
				double ap_o[3];
				s_as2ap(part.vs(), part.as(), contact_info[ic].pt, ap_o);
				s_inv_pm_dot_v3(contact_info[ic].T, ap_o, accel0.data() + ic * rows_per_contact + 3);
			}
		}
		return;
	}
	UniversalSolver::~UniversalSolver() = default;
	UniversalSolver::UniversalSolver(Size max_iter_count, double max_error) :Solver(max_iter_count, max_error) {}
	ARIS_DEFINE_BIG_FOUR_CPP(UniversalSolver);
#undef ARIS_LOOP_SYS
#undef ARIS_LOOP_SYS_D
#undef ARIS_LOOP_SYS_R
#undef ARIS_LOOP_BLOCK

	// Record activation states of prt jnt mot gm fce upon creation
	// And restore recorded states on destruction, to prevent state inconsistencies caused by modifications during method calls
	class HelpResetRAII{
	public:
		std::vector<bool> prt_active_, jnt_active_, mot_active_, gm_active_, fce_active_;
		Model *model_;

		HelpResetRAII(Model *model) : model_(model){
			for (auto &prt : model_->partPool())prt_active_.push_back(prt.active());
			for (auto &jnt : model_->jointPool())jnt_active_.push_back(jnt.active());
			for (auto &mot : model_->motionPool())mot_active_.push_back(mot.active());
			for (auto &gm : model_->generalMotionPool())gm_active_.push_back(gm.active());
			for (auto &fce : model_->forcePool())fce_active_.push_back(fce.active());
		}
		~HelpResetRAII(){
			for (auto &prt : model_->partPool())prt.activate(prt_active_[prt.id()]);
			for (auto &jnt : model_->jointPool())jnt.activate(jnt_active_[jnt.id()]);
			for (auto &mot : model_->motionPool())mot.activate(mot_active_[mot.id()]);
			for (auto &gm : model_->generalMotionPool())gm.activate(gm_active_[gm.id()]);
			for (auto &fce : model_->forcePool())fce.activate(fce_active_[fce.id()]);
		}
	};

	struct ForwardKinematicSolver::Imp { 
		double* J_{ nullptr }, * cf_{ nullptr };
		std::vector<char> mem_pool_;
		aris::Size mJf_{ 0 }, nJf_{ 0 };
	};
	auto ForwardKinematicSolver::allocateMemory()->void{
		HelpResetRAII help_reset(this->model());
		
		imp_->mJf_ = 0;
		imp_->nJf_ = model()->motionPool().size();

		for (auto &m : model()->motionPool())m.activate(true);
		for (auto& gm : model()->generalMotionPool()) {
			gm.activate(false);
			imp_->mJf_ += gm.dim();
		}

		aris::Size mem_pool_size{0};
		core::allocMem(mem_pool_size, imp_->J_, imp_->mJf_ * imp_->nJf_);
		core::allocMem(mem_pool_size, imp_->cf_, imp_->mJf_);

		imp_->mem_pool_.resize(mem_pool_size);

		imp_->J_ = core::getMem(imp_->mem_pool_.data(), imp_->J_);
		imp_->cf_ = core::getMem(imp_->mem_pool_.data(), imp_->cf_);

		UniversalSolver::allocateMemory();
	}
	
	auto ForwardKinematicSolver::kinPos()->int{
		UniversalSolver::kinPos();
		if (error() < maxError())for (auto &m : model()->generalMotionPool())m.updP();
		return error() < maxError() ? 0 : -1;
	}
	auto ForwardKinematicSolver::kinVel()->int{
		UniversalSolver::kinVel();
		for (auto &m : model()->generalMotionPool())m.updV();
		return 0;
	}
	auto ForwardKinematicSolver::dynAccAndFce()->int{
		UniversalSolver::dynAccAndFce();
		for (auto &m : model()->generalMotionPool())m.updA();
		return 0;
	}
	auto ForwardKinematicSolver::cptJacobi() noexcept->void{
		cptGeneralJacobi();

		// Need to derive velocities on each part caused by the end effector, then for the drive, find the velocity difference, this gives the velocity Jacobian, find the acceleration difference, which is cfi
		aris::Size pos = 0;
		for (auto &gm : model()->generalMotionPool()){
			for (auto &mot : model()->motionPool()){
				// reserve data //
				double vs_I_restore[6], vs_J_restore[6], as_I_restore[6], as_J_restore[6], mv_restore[6], ma_restore[6];
				gm.makI()->fatherPart().getVs(vs_I_restore);
				gm.makJ()->fatherPart().getVs(vs_J_restore);
				gm.makI()->fatherPart().getAs(as_I_restore);
				gm.makJ()->fatherPart().getAs(as_J_restore);
				gm.getV(mv_restore);
				gm.getA(ma_restore);
				
				// J //
				double vs_I[6], vs_J[6];
				s_vc(6, Jg() + at(gm.makI()->fatherPart().id() * 6, mot.id(), nJg()), nJg(), vs_I, 1);
				s_vc(6, Jg() + at(gm.makJ()->fatherPart().id() * 6, mot.id(), nJg()), nJg(), vs_J, 1);

				gm.makI()->fatherPart().setVs(vs_I);
				gm.makJ()->fatherPart().setVs(vs_J);
				gm.updV();
				s_vc(gm.dim(), gm.v(), 1, imp_->J_ + at(pos, mot.id(), nJf()), nJf());

				// restore vs //
				gm.makI()->fatherPart().setVs(vs_I_restore);
				gm.makJ()->fatherPart().setVs(vs_J_restore);
				gm.setV(mv_restore);

				// cf //
				double as_I[6], as_J[6];
				s_vc(6, cg() + gm.makI()->fatherPart().id() * 6, as_I);
				s_vc(6, cg() + gm.makJ()->fatherPart().id() * 6, as_J);
				gm.makI()->fatherPart().setAs(as_I);
				gm.makJ()->fatherPart().setAs(as_J);
				gm.updA();
				s_vc(gm.dim(), gm.a(), imp_->cf_ + pos);
				
				// restore data //
				gm.makI()->fatherPart().setAs(as_I_restore);
				gm.makJ()->fatherPart().setAs(as_J_restore);
				gm.setV(mv_restore);
				gm.setA(ma_restore);
			}
			pos += gm.dim();
		}
	}
	auto ForwardKinematicSolver::mJf()const noexcept->Size { return imp_->mJf_;  }
	auto ForwardKinematicSolver::nJf()const noexcept->Size { return imp_->nJf_; }
	auto ForwardKinematicSolver::Jf()const noexcept->const double * { return imp_->J_; }
	auto ForwardKinematicSolver::cf()const noexcept->const double * { return imp_->cf_; }
	ForwardKinematicSolver::~ForwardKinematicSolver() = default;
	ForwardKinematicSolver::ForwardKinematicSolver(Size max_iter_count, double max_error) :UniversalSolver(max_iter_count, max_error), imp_(new Imp) {}
	ARIS_DEFINE_BIG_FOUR_CPP(ForwardKinematicSolver);

	struct InverseKinematicSolver::Imp{
		double* J_{ nullptr }, * ci_{ nullptr };
		std::vector<char> mem_pool_;
		aris::Size mJi_{ 0 }, nJi_{ 0 };
	};
	auto InverseKinematicSolver::allocateMemory()->void{
		HelpResetRAII help_reset(this->model());
		
		imp_->mJi_ = model()->motionPool().size();
		imp_->nJi_ = 0;

		for (auto &m : model()->motionPool())m.activate(false);
		for (auto& gm : model()->generalMotionPool()) {
			gm.activate(true);
			imp_->nJi_ += gm.dim();
		}

		aris::Size mem_pool_size{ 0 };
		core::allocMem(mem_pool_size, imp_->J_, imp_->mJi_ * imp_->nJi_);
		core::allocMem(mem_pool_size, imp_->ci_, imp_->mJi_);

		imp_->mem_pool_.resize(mem_pool_size);

		imp_->J_ = core::getMem(imp_->mem_pool_.data(), imp_->J_);
		imp_->ci_ = core::getMem(imp_->mem_pool_.data(), imp_->ci_);

		UniversalSolver::allocateMemory();
	}
	auto InverseKinematicSolver::kinPos()->int{
		UniversalSolver::kinPos();
		if (error() < maxError())for (auto &m : model()->motionPool())m.updP();
		return error() < maxError() ? 0 : -1;
	}
	auto InverseKinematicSolver::kinVel()->int{
		UniversalSolver::kinVel();
		for (auto &m : model()->motionPool())m.updV();
		return 0;
	}
	auto InverseKinematicSolver::dynAccAndFce()->int{
		UniversalSolver::dynAccAndFce();
		for (auto &m : model()->motionPool())m.updA();
		return 0;
	}
	auto InverseKinematicSolver::cptJacobi()noexcept->void{
		cptGeneralJacobi();

		// Need to derive velocities on each part caused by the end effector, then for the drive, find the velocity difference, this gives the velocity Jacobian
		aris::Size pos = 0;
		for (auto &gm : model()->generalMotionPool()){
			for (auto &mot : model()->motionPool()){
				for (Size i = 0; i < gm.dim(); ++i) {
					double tem[6], tem2[6];
					s_vc(6, Jg() + at(mot.makI()->fatherPart().id() * 6, gm.id() * 6 + i, nJg()), nJg(), tem, 1);
					s_vs(6, Jg() + at(mot.makJ()->fatherPart().id() * 6, gm.id() * 6 + i, nJg()), nJg(), tem, 1);

					s_inv_tv(*mot.makI()->pm(), tem, tem2);
					imp_->J_[at(mot.id(), pos + i, nJi())] = tem2[mot.axis()];

					// Solve ci below //
					// This section is equivalent to updMv //
					mot.makI()->getVs(*mot.makJ(), tem);
					double dq = tem[mot.axis()];
					s_cv(mot.makJ()->vs(), mot.makI()->vs(), tem2);

					// ai - aj - vi x (vi - vj) * dq = ai - aj - v1 x v2 * dq
					s_vc(6, cg() + mot.makI()->fatherPart().id() * 6, tem);
					s_vs(6, cg() + mot.makJ()->fatherPart().id() * 6, tem);
					s_va(6, -dq, tem2, tem);
					s_inv_tv(*mot.makI()->pm(), tem, tem2);
					imp_->ci_[mot.id()] = tem2[mot.axis()];
				}
			}
			pos += gm.dim();
		}
	}
	auto InverseKinematicSolver::mJi()const noexcept->Size { return imp_->mJi_; }
	auto InverseKinematicSolver::nJi()const noexcept->Size { return imp_->nJi_; }
	auto InverseKinematicSolver::Ji()const noexcept->const double * { return imp_->J_; }
	auto InverseKinematicSolver::ci()const noexcept->const double * { return imp_->ci_; }
	InverseKinematicSolver::~InverseKinematicSolver() = default;
	InverseKinematicSolver::InverseKinematicSolver(Size max_iter_count, double max_error) :UniversalSolver(max_iter_count, max_error), imp_(new Imp) {}
	ARIS_DEFINE_BIG_FOUR_CPP(InverseKinematicSolver);

	struct ForwardDynamicSolver::Imp{
		double* MBeta_{ nullptr }, * ci_{ nullptr };
		std::vector<char> mem_pool_;
		aris::Size mJi_{ 0 }, nJi_{ 0 };
	};
	auto ForwardDynamicSolver::allocateMemory()->void{
		HelpResetRAII help_reset(this->model());
		
		for (auto &m : model()->motionPool())m.activate(false);
		for (auto &gm : model()->generalMotionPool())gm.activate(false);
		for (auto &f : model()->forcePool())f.activate(true);
		UniversalSolver::allocateMemory();
	}
	auto ForwardDynamicSolver::kinPos()->int{
		UniversalSolver::kinPos();
		if (error() < maxError())for (auto &m : model()->generalMotionPool())m.updP();
		return error() < maxError() ? 0 : -1;
	}
	auto ForwardDynamicSolver::kinVel()->int
	{
		UniversalSolver::kinVel();
		for (auto &m : model()->generalMotionPool())m.updV();
		return 0;
	}
	auto ForwardDynamicSolver::dynAccAndFce()->int
	{
		UniversalSolver::dynAccAndFce();
		for (auto &m : model()->generalMotionPool())m.updA();
		return 0;
	}
	auto ForwardDynamicSolver::cptProjectedMassMatrix() noexcept -> void {
		UniversalSolver::cptProjectedMassMatrix();
		return;
	}
	auto ForwardDynamicSolver::cptContactInverseInertiaMatrix(int nContact, int* partid, 
		double* T_vec, double* contactPoint, std::vector<double>& A_out, std::vector<double>& accel0) noexcept -> void {
		UniversalSolver::cptContactInverseInertiaMatrix(nContact, partid, T_vec, contactPoint, A_out, accel0);
		return;
	}
	ForwardDynamicSolver::~ForwardDynamicSolver() = default;
	ForwardDynamicSolver::ForwardDynamicSolver(Size max_iter_count, double max_error) :UniversalSolver(max_iter_count, max_error) {}
	ARIS_DEFINE_BIG_FOUR_CPP(ForwardDynamicSolver);

	auto InverseDynamicSolver::allocateMemory()->void
	{
		HelpResetRAII help_reset(this->model());
		
		for (auto &m : model()->motionPool())m.activate(true);
		for (auto &gm : model()->generalMotionPool())gm.activate(false);
		for (auto &f : model()->forcePool())f.activate(false);
		UniversalSolver::allocateMemory();
	}
	auto InverseDynamicSolver::kinPos()->int
	{
		UniversalSolver::kinPos();
		if (error() < maxError())for (auto &m : model()->motionPool())m.updP();
		return error() < maxError() ? 0 : -1;
	}
	auto InverseDynamicSolver::kinVel()->int
	{
		UniversalSolver::kinVel();
		for (auto &m : model()->motionPool())m.updV();
		return 0;
	}
	auto InverseDynamicSolver::dynAccAndFce()->int
	{
		UniversalSolver::dynAccAndFce();
		for (auto &m : model()->generalMotionPool())m.updA();
		return 0;
	}
	InverseDynamicSolver::~InverseDynamicSolver() = default;
	InverseDynamicSolver::InverseDynamicSolver(Size max_iter_count, double max_error) :UniversalSolver(max_iter_count, max_error) {}
	ARIS_DEFINE_BIG_FOUR_CPP(InverseDynamicSolver);

	ARIS_REGISTRATION{
		aris::core::class_<Solver>("Solver")
			.prop("root_num", &Solver::setRootNumber, &Solver::rootNumber)
			.prop("which_root", &Solver::setWhichRoot, &Solver::whichRoot)
			.prop("max_iter_count", &Solver::setMaxIterCount, &Solver::maxIterCount)
			.prop("max_error", &Solver::setMaxError, &Solver::maxError)
			;

		aris::core::class_<UniversalSolver>("UniversalSolver")
			.inherit<Solver>()
			;

		aris::core::class_<ForwardKinematicSolver>("ForwardKinematicSolver")
			.inherit<UniversalSolver>()
			;

		aris::core::class_<InverseKinematicSolver>("InverseKinematicSolver")
			.inherit<UniversalSolver>()
			;

		aris::core::class_<ForwardDynamicSolver>("ForwardDynamicSolver")
			.inherit<UniversalSolver>()
			;

		aris::core::class_<InverseDynamicSolver>("InverseDynamicSolver")
			.inherit<UniversalSolver>()
			;
	}
}
