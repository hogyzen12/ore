use anyhow::Result;
use atlas::{Atlas, resolve_token};
use ore_api::prelude::*;
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::{
    compute_budget::ComputeBudgetInstruction,
    signature::{read_keypair_file, Keypair, Signer},
    transaction::Transaction,
};
use steel::{AccountDeserialize, Clock};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::sleep;

// ========== CONFIG ==========
const REF_MULT: f64 = 0.9;
const ADMIN_FEE: f64 = 0.01;
const PROTOCOL_CUT: f64 = 0.10;
const P_WIN: f64 = 1.0 / 25.0;
const HIT_PROB: f64 = 1.0 / 625.0;
const UPDATE_INTERVAL_MS: u64 = 2000; // Check every 2 seconds (less spam, still responsive)
const RESET_COOLDOWN_MS: u64 = 5000;
const MIN_TOTAL_SOL: f64 = 0.1;
const MIN_EV_THRESHOLD: f64 = 0.0; // Any positive EV

// === SIMPLE BETTING STRATEGY CONFIG ===
// Place one bet per round:
// - Base coverage: All 25 squares
// - EV boost: Top 10 highest EV squares
// Target: 0.015 SOL per round (sustainable)

// === BET CONFIGURATION ===
// Base coverage: Dynamically calculated from average +EV per square
// This adapts to pool conditions automatically

// Minimum base bet (safety floor)
const MIN_BASE_BET: f64 = 0.0001; // Never bet less than this per square
const MAX_BASE_BET: f64 = 0.001;  // Cap base bet for safety

// EV boost: Pool-size-based percentages (added on top of base)
const MIN_ROI_THRESHOLD: f64 = 1.0; // Only boost squares with >1% ROI

// Boost as percentage of square's deployed pool
const BOOST_SMALL_EDGE_PCT: f64 = 0.01;  // 1% of pool for 1-2% ROI
const BOOST_MEDIUM_EDGE_PCT: f64 = 0.02; // 2% of pool for >2% ROI

// Total per round: variable based on pool conditions and sizes

// === TIMING ===
// Round duration: 150 slots = 60 seconds
const ROUND_DURATION_SLOTS: u64 = 150;

// Bet placement: Deploy after round starts
const BET_AFTER_START_SLOTS: u64 = 110; // Deploy 110 slots (~44 sec) into round

// === REWARD CLAIMING CONFIG ===
// Claim rewards when SOL balance exceeds this threshold (to keep wallet topped up)
const MIN_SOL_TO_CLAIM: f64 = 0.001; // Claim if we have > 0.001 SOL rewards
const MIN_ORE_TO_CLAIM: f64 = 0.01;  // Claim if we have > 0.01 ORE rewards

// Derived constants
const ADMIN_COST_FACTOR: f64 = ADMIN_FEE / (1.0 - ADMIN_FEE);
const C: f64 = 24.0 + (ADMIN_COST_FACTOR / P_WIN);

// ========== STATE ==========
struct BotState {
    price_ore_sol: Arc<RwLock<f64>>,
    last_round_id: Arc<RwLock<Option<u64>>>,
    last_round_reset: Arc<RwLock<Option<Instant>>>,
    is_in_cooldown: Arc<RwLock<bool>>,
}

impl BotState {
    fn new() -> Self {
        Self {
            price_ore_sol: Arc::new(RwLock::new(0.55)), // Fallback: 110/200
            last_round_id: Arc::new(RwLock::new(None)),
            last_round_reset: Arc::new(RwLock::new(None)),
            is_in_cooldown: Arc::new(RwLock::new(false)),
        }
    }
}

// ========== HELPERS ==========
fn lamports_to_sol(lamports: u64) -> f64 {
    lamports as f64 / 1_000_000_000.0
}

fn sol_to_lamports(sol: f64) -> u64 {
    (sol * 1_000_000_000.0) as u64
}

// ========== EV CALCULATION ==========
#[derive(Debug, Clone)]
struct SquareEV {
    square_id: usize,
    deployed_sol: f64,
    optimal_bet_sol: f64,
    ev_sol: f64,
}

fn compute_ev_star_for_square(
    o: f64,       // SOL deployed on this square
    t: f64,       // Total SOL deployed across all squares
    ore_value_in_sol: f64,
) -> (f64, f64) {
    if !o.is_finite() || o <= 0.0 || !t.is_finite() || t <= 0.0 {
        return (0.0, f64::NEG_INFINITY);
    }

    let v_initial = (1.0 - PROTOCOL_CUT) * (t - o) + ore_value_in_sol;
    if v_initial <= 0.0 {
        return (0.0, f64::NEG_INFINITY);
    }

    // Initial guess
    let mut y_star = ((ore_value_in_sol * o) / C).sqrt();
    let mut v = v_initial;

    // Iterate to refine (3 iterations like the JS script)
    for _ in 0..3 {
        v = (1.0 - PROTOCOL_CUT) * (t - o - y_star) + ore_value_in_sol;
        if v <= 0.0 {
            break;
        }
        let new_y = ((v * o) / C).sqrt() - o;
        y_star = new_y.max(0.0);
    }

    if y_star <= 0.0 {
        return (0.0, 0.0);
    }

    let f = y_star / (o + y_star);
    let admin_cost = ADMIN_COST_FACTOR * y_star;
    let ev = P_WIN * (-24.0 * y_star + v * f) - admin_cost;

    (y_star, ev)
}

fn calculate_all_ev(
    deployed: &[u64; 25],
    ore_price_sol: f64,
    motherlode_ore: f64,
) -> Vec<SquareEV> {
    // Calculate total deployed
    let total_sol: f64 = deployed.iter().map(|&d| lamports_to_sol(d)).sum();

    if total_sol < MIN_TOTAL_SOL {
        return vec![];
    }

    // Calculate ORE value with motherlode expectation
    let expected_motherlode = motherlode_ore * HIT_PROB;
    let ore_value_in_sol = ore_price_sol * REF_MULT * (1.0 + expected_motherlode);

    let mut results = Vec::new();

    for (square_id, &deployed_lamports) in deployed.iter().enumerate() {
        let deployed_sol = lamports_to_sol(deployed_lamports);

        if deployed_sol <= 0.0 {
            continue;
        }

        let (y_star, ev) = compute_ev_star_for_square(deployed_sol, total_sol, ore_value_in_sol);

        if ev.is_finite() {
            results.push(SquareEV {
                square_id,
                deployed_sol,
                optimal_bet_sol: y_star,
                ev_sol: ev,
            });
        }
    }

    results
}

// ========== SOLANA FETCHERS ==========
async fn get_board(rpc: &RpcClient) -> Result<Board> {
    let board_pda = ore_api::state::board_pda();
    let account = rpc.get_account(&board_pda.0).await?;
    let board = Board::try_from_bytes(&account.data)?;
    Ok(*board)
}

async fn get_round(rpc: &RpcClient, id: u64) -> Result<Round> {
    let round_pda = ore_api::state::round_pda(id);
    let account = rpc.get_account(&round_pda.0).await?;
    let round = Round::try_from_bytes(&account.data)?;
    Ok(*round)
}

async fn get_miner(rpc: &RpcClient, authority: solana_sdk::pubkey::Pubkey) -> Result<Miner> {
    let miner_pda = ore_api::state::miner_pda(authority);
    let account = rpc.get_account(&miner_pda.0).await?;
    let miner = Miner::try_from_bytes(&account.data)?;
    Ok(*miner)
}

async fn get_clock(rpc: &RpcClient) -> Result<Clock> {
    let data = rpc.get_account_data(&solana_sdk::sysvar::clock::ID).await?;
    let clock = bincode::deserialize::<Clock>(&data)?;
    Ok(clock)
}

async fn get_treasury(rpc: &RpcClient) -> Result<Treasury> {
    let treasury_pda = ore_api::state::treasury_pda();
    let account = rpc.get_account(&treasury_pda.0).await?;
    let treasury = Treasury::try_from_bytes(&account.data)?;
    Ok(*treasury)
}

// ========== TRANSACTION HELPERS ==========
async fn submit_transaction(
    rpc: &RpcClient,
    payer: &Keypair,
    instructions: &[solana_sdk::instruction::Instruction],
) -> Result<String> {
    let blockhash = rpc.get_latest_blockhash().await?;
    let mut all_instructions = vec![
        ComputeBudgetInstruction::set_compute_unit_limit(1_400_000),
        ComputeBudgetInstruction::set_compute_unit_price(1_000_000),
    ];
    all_instructions.extend_from_slice(instructions);

    let transaction = Transaction::new_signed_with_payer(
        &all_instructions,
        Some(&payer.pubkey()),
        &[payer],
        blockhash,
    );

    let signature = rpc.send_and_confirm_transaction(&transaction).await?;
    Ok(signature.to_string())
}

async fn execute_checkpoint(
    rpc: &RpcClient,
    payer: &Keypair,
    authority: solana_sdk::pubkey::Pubkey,
) -> Result<()> {
    println!("⏳ Checkpointing previous round...");

    let miner = get_miner(rpc, authority).await?;

    // Only checkpoint if needed
    if miner.checkpoint_id < miner.round_id {
        let ix = ore_api::sdk::checkpoint(payer.pubkey(), authority, miner.round_id);
        let sig = submit_transaction(rpc, payer, &[ix]).await?;
        println!("  ✓ Checkpoint complete: {}", sig);
    } else {
        println!("  ✓ Already checkpointed");
    }

    Ok(())
}

async fn check_and_claim_rewards(
    rpc: &RpcClient,
    payer: &Keypair,
) -> Result<()> {
    // Get miner account to check pending rewards
    let miner = match get_miner(rpc, payer.pubkey()).await {
        Ok(m) => m,
        Err(_) => return Ok(()), // No miner account yet
    };

    let rewards_sol = lamports_to_sol(miner.rewards_sol);
    let rewards_ore = miner.rewards_ore as f64 / 100_000_000_000.0; // 11 decimals

    // Check if we have rewards worth claiming
    if rewards_sol >= MIN_SOL_TO_CLAIM || rewards_ore >= MIN_ORE_TO_CLAIM {
        println!("\n💰 Claiming Rewards!");
        println!("  Pending SOL: {:.6} SOL", rewards_sol);
        println!("  Pending ORE: {:.6} ORE", rewards_ore);

        // Create claim instructions
        let ix_sol = ore_api::sdk::claim_sol(payer.pubkey());
        let ix_ore = ore_api::sdk::claim_ore(payer.pubkey());

        let sig = submit_transaction(rpc, payer, &[ix_sol, ix_ore]).await?;

        println!("  ✓ Rewards claimed!");
        println!("  📋 Transaction: https://solscan.io/tx/{}", sig);
        println!("  💵 Wallet topped up with {:.6} SOL\n", rewards_sol);
    }

    Ok(())
}

async fn execute_deploy_multiple(
    rpc: &RpcClient,
    payer: &Keypair,
    board: &Board,
    square_amounts: &[(usize, f64)], // Vec of (square_id, amount_sol)
) -> Result<String> {
    // Group squares by amount (so we can batch by amount)
    let mut amount_groups: std::collections::HashMap<u64, Vec<usize>> = std::collections::HashMap::new();

    for (square_id, amount_sol) in square_amounts {
        let lamports = sol_to_lamports(*amount_sol);
        amount_groups.entry(lamports).or_insert_with(Vec::new).push(*square_id);
    }

    // Create deploy instructions for each amount group
    let mut instructions = vec![];
    for (amount_lamports, square_ids) in amount_groups.iter() {
        let mut squares = [false; 25];
        for &square_id in square_ids {
            squares[square_id] = true;
        }

        let ix = ore_api::sdk::deploy(
            payer.pubkey(),
            payer.pubkey(),
            *amount_lamports,
            board.round_id,
            squares,
        );
        instructions.push(ix);
    }

    let sig = submit_transaction(rpc, payer, &instructions).await?;
    Ok(sig)
}

// ========== PRICE STREAMING ==========
async fn start_price_stream(state: Arc<BotState>) -> Result<()> {
    println!("🔄 Starting ORE/SOL price stream...");

    let atlas = Atlas::new().await?;

    // Get token mints and decimals
    let ore_mint = resolve_token("ORE").map_err(|e| anyhow::anyhow!(e))?;
    let sol_mint = "So11111111111111111111111111111111111111112"; // Wrapped SOL

    // Get proper decimals from tokens.json
    let ore_decimals = atlas::get_token_decimals("ORE")
        .ok_or_else(|| anyhow::anyhow!("Could not find ORE decimals"))?;
    let sol_decimals = atlas::get_token_decimals("SOL")
        .ok_or_else(|| anyhow::anyhow!("Could not find SOL decimals"))?;

    println!("  Token decimals: ORE={}, SOL={}", ore_decimals, sol_decimals);

    // Stream quotes for 1 ORE → SOL (using correct decimals)
    let amount = 10_u64.pow(ore_decimals as u32); // 1 ORE with proper decimals
    let mut stream = atlas.quote_stream(&ore_mint, sol_mint, amount).await?;

    println!("  ✓ Price stream connected (stream #{})", stream.stream_id());

    tokio::spawn(async move {
        while let Some(quotes) = stream.next().await {
            // Get the best quote from the HashMap
            if let Some((_provider, best_quote)) = quotes.quotes.iter().next() {
                // Convert output from lamports to SOL using proper decimals
                let ore_to_sol_price = best_quote.out_amount as f64 / 10_f64.powi(sol_decimals as i32);
                *state.price_ore_sol.write().await = ore_to_sol_price;

                // Log occasionally (every ~10 updates)
                if rand::random::<u8>() < 25 {
                    println!("  💰 ORE/SOL price: {:.6}", ore_to_sol_price);
                }
            }
        }
    });

    Ok(())
}

// ========== LOGGING HELPERS ==========
fn log_pool_state(round: &Round, clock: &Clock, board: &Board) {
    let total_sol: f64 = round.deployed.iter().map(|&d| lamports_to_sol(d)).sum();
    let slots_since_start = clock.slot.saturating_sub(board.start_slot);
    let slots_until_end = board.end_slot.saturating_sub(clock.slot);
    
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 POOL STATE - Round #{}", board.round_id);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("⏱️  Round Progress:");
    println!("   Slot: {} / {} (start: {})", clock.slot, board.end_slot, board.start_slot);
    println!("   Elapsed: {} slots ({:.1}s)", slots_since_start, slots_since_start as f64 * 0.4);
    println!("   Remaining: {} slots ({:.1}s)", slots_until_end, slots_until_end as f64 * 0.4);
    println!();
    println!("💰 Total Pool: {:.4} SOL across 25 squares", total_sol);
    println!("   Average per square: {:.4} SOL", total_sol / 25.0);
    println!();
    println!("📍 Deployments by Square:");
    for i in 0..25 {
        let deployed = lamports_to_sol(round.deployed[i]);
        let miners = round.count[i];
        if i % 5 == 0 && i > 0 {
            println!();
        }
        print!("   #{:2}: {:.4} SOL ({:2} miners)  ", i, deployed, miners);
    }
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

fn log_ev_analysis(ev_results: &[SquareEV], ore_price_sol: f64, motherlode_ore: f64, top_n: usize) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🧮 EV CALCULATION ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📈 Market Data:");
    println!("   ORE/SOL Price: {:.6} SOL", ore_price_sol);
    println!("   Motherlode: {:.2} ORE ({:.6} SOL expected value)", 
             motherlode_ore, motherlode_ore * HIT_PROB * ore_price_sol);
    println!("   ORE Value (with motherlode): {:.6} SOL", 
             ore_price_sol * REF_MULT * (1.0 + motherlode_ore * HIT_PROB));
    println!();
    
    let mut sorted = ev_results.to_vec();
    sorted.sort_by(|a, b| b.ev_sol.partial_cmp(&a.ev_sol).unwrap());
    
    println!("🎯 Top {} EV Squares:", top_n);
    println!("   Rank | Square | Deployed | Optimal Bet | EV (SOL) | ROI");
    println!("   -----|--------|----------|-------------|----------|--------");
    for (i, square) in sorted.iter().take(top_n).enumerate() {
        let roi = if square.optimal_bet_sol > 0.0 {
            (square.ev_sol / square.optimal_bet_sol) * 100.0
        } else {
            0.0
        };
        println!("   {:4} | #{:5} | {:8.4} | {:11.6} | {:8.6} | {:6.1}%",
                 i + 1, square.square_id, square.deployed_sol, 
                 square.optimal_bet_sol, square.ev_sol, roi);
    }
    
    let positive_ev_count = ev_results.iter().filter(|s| s.ev_sol > 0.0).count();
    let total_positive_ev: f64 = ev_results.iter()
        .filter(|s| s.ev_sol > 0.0)
        .map(|s| s.ev_sol)
        .sum();
    
    println!();
    println!("📊 EV Summary:");
    println!("   Squares with +EV: {}/25", positive_ev_count);
    println!("   Total +EV available: {:.6} SOL", total_positive_ev);
    println!("   Average +EV per square: {:.6} SOL", 
             if positive_ev_count > 0 { total_positive_ev / positive_ev_count as f64 } else { 0.0 });
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

fn log_bet_placement(square_amounts: &[(usize, f64)], ev_results: &[SquareEV], base_bet: f64) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("🎯 BET PLACEMENT BREAKDOWN");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let base_total = base_bet * 25.0;
    let grand_total: f64 = square_amounts.iter().map(|(_, amt)| amt).sum();
    let boost_total = grand_total - base_total;
    
    println!("💵 Cost Summary:");
    println!("   Base coverage (25 squares × {:.6} SOL): {:.6} SOL", 
             base_bet, base_total);
    println!("   EV boost (dynamic ROI-based):           {:.4} SOL", boost_total);
    println!("   ─────────────────────────────────────────");
    println!("   TOTAL BET:                               {:.4} SOL", grand_total);
    println!();
    
    // Collect and sort boosted squares
    let mut boosted_squares: Vec<_> = square_amounts.iter()
        .filter(|(_, amt)| *amt > base_bet)
        .collect();
    boosted_squares.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    if !boosted_squares.is_empty() {
        println!("🎲 ROI-Based Boosted Bets:");
        for (i, &&(square_id, total_bet)) in boosted_squares.iter().enumerate() {
            if let Some(square_ev) = ev_results.iter().find(|s| s.square_id == square_id) {
                let boost = total_bet - base_bet;
                let roi = if square_ev.optimal_bet_sol > 0.0 {
                    (square_ev.ev_sol / square_ev.optimal_bet_sol) * 100.0
                } else {
                    0.0
                };
                println!("   {:2}. Square #{:2}: {:.6} SOL ({:.6} base + {:.4} boost) | ROI: {:5.1}% | EV: {:+.6} SOL",
                         i + 1, square_id, total_bet, base_bet, boost, roi, square_ev.ev_sol);
            }
        }
    }
    
    println!();
    println!("📋 Bet Distribution:");
    println!("   {}/25 squares: {:.6} SOL (base only)", 
             25 - boosted_squares.len(), base_bet);
    println!("   {}/25 squares: Dynamic boost based on ROI", boosted_squares.len());
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

// ========== MAIN BOT LOOP ==========
async fn bot_tick(
    state: Arc<BotState>,
    rpc: &RpcClient,
    payer: &Keypair,
) -> Result<()> {
    // Get board state
    let board = get_board(rpc).await?;
    let round = get_round(rpc, board.round_id).await?;
    let clock = get_clock(rpc).await?;

    // Check for round reset
    {
        let mut last_round = state.last_round_id.write().await;
        if let Some(prev_round) = *last_round {
            if board.round_id != prev_round {
                println!("\n╔════════════════════════════════════════════════════╗");
                println!("║  🔄 ROUND TRANSITION                              ║");
                println!("╚════════════════════════════════════════════════════╝");
                println!("   Previous Round: #{}", prev_round);
                println!("   New Round:      #{}", board.round_id);
                println!("   Entering cooldown period ({:.1}s)", RESET_COOLDOWN_MS as f64 / 1000.0);
                
                *state.last_round_reset.write().await = Some(Instant::now());
                *state.is_in_cooldown.write().await = true;
                *last_round = Some(board.round_id);

                // Checkpoint previous round before proceeding
                execute_checkpoint(rpc, payer, payer.pubkey()).await?;

                // Claim rewards after checkpoint (keeps SOL topped up)
                check_and_claim_rewards(rpc, payer).await?;

                return Ok(());
            }
        } else {
            println!("\n╔════════════════════════════════════════════════════╗");
            println!("║  🎬 BOT STARTED - Round #{}                       ║", board.round_id);
            println!("╚════════════════════════════════════════════════════╝");
            
            // On startup, checkpoint if needed before doing anything else
            execute_checkpoint(rpc, payer, payer.pubkey()).await?;
            
            // Also claim any pending rewards
            check_and_claim_rewards(rpc, payer).await?;
            
            *last_round = Some(board.round_id);
        }
    }

    // Check cooldown
    {
        let is_cooldown = *state.is_in_cooldown.read().await;
        if is_cooldown {
            if let Some(reset_time) = *state.last_round_reset.read().await {
                let elapsed = reset_time.elapsed();
                if elapsed < Duration::from_millis(RESET_COOLDOWN_MS) {
                    return Ok(()); // Still in cooldown
                } else {
                    println!("  ✓ Cooldown complete for Round #{}", board.round_id);
                    *state.is_in_cooldown.write().await = false;
                }
            }
        }
    }

    // === SIMPLE BETTING STRATEGY ===
    // Check if we've deployed in THIS round (not previous)
    let has_bet = match get_miner(rpc, payer.pubkey()).await {
        Ok(miner) => {
            // CRITICAL: Check if we have bet placed
            // After checkpoint, miner.round_id updates to the new round, but deployed array
            // still has old values. If checkpoint_id == round_id, we've checkpointed but not
            // deployed yet in the new round.
            if miner.round_id != board.round_id || 
               miner.deployed.iter().all(|&d| d == 0) ||
               miner.checkpoint_id == miner.round_id {
                // Not deployed in current round yet
                false
            } else {
                // We have deployment in this round - bet has been placed
                // Bet puts at least MIN_BASE_BET on all squares
                miner.deployed.iter().all(|&d| lamports_to_sol(d) >= MIN_BASE_BET * 0.8)
            }
        }
        Err(_) => {
            // No miner account yet, will be created on first deploy
            false
        }
    };

    let slots_since_start = clock.slot.saturating_sub(board.start_slot);
    let slots_until_end = board.end_slot.saturating_sub(clock.slot);
    let seconds_since_start = slots_since_start as f64 * 0.4;
    let seconds_until_end = slots_until_end as f64 * 0.4;

    // Get current price and motherlode for EV calculations
    let ore_price_sol = *state.price_ore_sol.read().await;
    let treasury = get_treasury(rpc).await?;
    let motherlode_ore = treasury.motherlode as f64 / 100_000_000_000.0;

    // Calculate EV for all squares
    let ev_results = calculate_all_ev(&round.deployed, ore_price_sol, motherlode_ore);

    if ev_results.is_empty() {
        println!("⚠️  No EV data available (pool too small: < {:.2} SOL)", MIN_TOTAL_SOL);
        return Ok(());
    }

    // === PLACE BET (Once per round) ===
    if !has_bet {
        // Wait until we're past the initial slots
        if slots_since_start < BET_AFTER_START_SLOTS {
            if slots_since_start % 10 == 0 {  // Log occasionally
                println!("⏳ Round #{}: Waiting to place bet... ({:.1}s elapsed, {:.1}s remaining until bet window)",
                         board.round_id, seconds_since_start, 
                         (BET_AFTER_START_SLOTS - slots_since_start) as f64 * 0.4);
            }
            return Ok(());
        }

        // Log pool state before betting
        log_pool_state(&round, &clock, &board);

        // Log EV analysis
        log_ev_analysis(&ev_results, ore_price_sol, motherlode_ore, 15);

        // Calculate dynamic base bet from average +EV per square
        let positive_ev_count = ev_results.iter().filter(|s| s.ev_sol > 0.0).count();
        let total_positive_ev: f64 = ev_results.iter()
            .filter(|s| s.ev_sol > 0.0)
            .map(|s| s.ev_sol)
            .sum();
        
        let base_bet = if positive_ev_count > 0 {
            let avg_ev = total_positive_ev / positive_ev_count as f64;
            // Clamp between min and max for safety
            avg_ev.max(MIN_BASE_BET).min(MAX_BASE_BET)
        } else {
            MIN_BASE_BET
        };

        println!("📏 Dynamic Base Bet: {:.6} SOL (avg +EV per square)\n", base_bet);

        // Build bet deployment with pool-size-based boost
        let mut square_amounts: Vec<(usize, f64)> = Vec::new();
        for square_id in 0..25 {
            // Check if this square has +EV and calculate pool-based boost
            let boost = if let Some(square_ev) = ev_results.iter().find(|s| s.square_id == square_id) {
                if square_ev.optimal_bet_sol > 0.0 {
                    let roi = (square_ev.ev_sol / square_ev.optimal_bet_sol) * 100.0;
                    
                    // Calculate boost as percentage of square's deployed pool
                    if roi > 2.0 {
                        // >2% ROI: bet 2% of pool
                        square_ev.deployed_sol * BOOST_MEDIUM_EDGE_PCT
                    } else if roi > MIN_ROI_THRESHOLD {
                        // 1-2% ROI: bet 1% of pool
                        square_ev.deployed_sol * BOOST_SMALL_EDGE_PCT
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };
            
            square_amounts.push((square_id, base_bet + boost));
        }

        // Log bet placement details
        log_bet_placement(&square_amounts, &ev_results, base_bet);

        // Execute deployment
        println!("🚀 Executing deployment transaction...");
        let sig = execute_deploy_multiple(rpc, payer, &board, &square_amounts).await?;
        
        let total_deployed: f64 = square_amounts.iter().map(|(_, amt)| amt).sum();
        
        println!("\n╔════════════════════════════════════════════════════╗");
        println!("║  ✅ BET PLACED SUCCESSFULLY                        ║");
        println!("╚════════════════════════════════════════════════════╝");
        println!("📋 Transaction: https://solscan.io/tx/{}", sig);
        println!("💰 Total Deployed: {:.4} SOL", total_deployed);
        println!("⏱️  Time in round: {:.1}s / {:.1}s\n", seconds_since_start, 
                 ROUND_DURATION_SLOTS as f64 * 0.4);

        return Ok(());
    }

    // Bet already placed this round - wait for next round
    if slots_since_start % 20 == 0 {  // Log occasionally
        println!("✓ Round #{}: Bet already placed, waiting for round end ({:.1}s remaining)",
                 board.round_id, seconds_until_end);
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("\n🤖 ORE EV Bot Starting...\n");

    // Load config
    let keypair_path = std::env::var("KEYPAIR")
        .expect("KEYPAIR env var required (path to keypair file)");
    let rpc_url = std::env::var("RPC")
        .expect("RPC env var required (Solana RPC URL)");

    let payer = read_keypair_file(&keypair_path)
        .map_err(|e| anyhow::anyhow!("Failed to read keypair: {}", e))?;
    let rpc = RpcClient::new(rpc_url);

    println!("📍 Wallet: {}", payer.pubkey());

    // Check wallet balance
    let wallet_balance = rpc.get_balance(&payer.pubkey()).await?;
    let wallet_sol = lamports_to_sol(wallet_balance);
    println!("💵 Wallet balance: {:.6} SOL", wallet_sol);

    // Calculate estimated total per round (dynamic base + typical EV boosts)
    let total_per_round = 0.005 + 0.010; // Estimate: ~0.005 SOL base + ~0.010 SOL in boosts

    // Calculate recommended minimum
    let recommended_min = total_per_round * 10.0; // 10 rounds buffer
    if wallet_sol < recommended_min {
        println!("⚠️  WARNING: Low balance! Recommended minimum: {:.2} SOL", recommended_min);
        println!("   Your balance: {:.6} SOL", wallet_sol);
    } else {
        println!("✅ Balance sufficient for ~{:.0} rounds", wallet_sol / total_per_round);
    }

    // Initialize state
    let state = Arc::new(BotState::new());

    // Start price stream
    start_price_stream(Arc::clone(&state)).await?;

    // Wait for initial price
    sleep(Duration::from_secs(2)).await;

    // Display initial miner stats if available
    if let Ok(miner) = get_miner(&rpc, payer.pubkey()).await {
        println!("\n📊 Miner Account Found!");
        println!("  Lifetime SOL rewards: {:.6} SOL", lamports_to_sol(miner.lifetime_rewards_sol));
        println!("  Lifetime ORE rewards: {:.6} ORE", miner.lifetime_rewards_ore as f64 / 100_000_000_000.0);
        println!("  Pending SOL: {:.6} SOL", lamports_to_sol(miner.rewards_sol));
        println!("  Pending ORE: {:.6} ORE", miner.rewards_ore as f64 / 100_000_000_000.0);
        println!("  Current round: #{}", miner.round_id);
        println!("  Checkpoint: #{}", miner.checkpoint_id);
    }

    println!("\n🎮 Bot running with SIMPLE BETTING Strategy!\n");
    println!("Strategy:");
    println!("  Round duration: {} slots (~{} seconds)", ROUND_DURATION_SLOTS, (ROUND_DURATION_SLOTS as f64 * 0.4) as u64);
    println!("  Bet placed: ~{}s into each round", (BET_AFTER_START_SLOTS as f64 * 0.4) as u64);
    println!();
    println!("  Base coverage: Dynamic (avg +EV per square)");
    println!("    Range: {:.6} - {:.6} SOL per square", MIN_BASE_BET, MAX_BASE_BET);
    println!("  EV boost: Pool-size-based (added on top of base)");
    println!("    >2% ROI: 2% of square's pool | 1-2% ROI: 1% of square's pool");
    println!();
    println!("  Estimated total per round: ~{:.4} SOL", total_per_round);
    println!("  100% participation guaranteed");
    println!("  Check interval: {}ms\n", UPDATE_INTERVAL_MS);

    // Main loop
    loop {
        match bot_tick(Arc::clone(&state), &rpc, &payer).await {
            Ok(_) => {}
            Err(e) => {
                eprintln!("❌ Error in bot tick: {}", e);
            }
        }

        sleep(Duration::from_millis(UPDATE_INTERVAL_MS)).await;
    }
}
