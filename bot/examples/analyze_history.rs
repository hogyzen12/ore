use anyhow::Result;
use clap::Parser;
use ore_api::prelude::*;
use solana_client::nonblocking::rpc_client::RpcClient;
use serde::{Deserialize, Serialize};
use steel::AccountDeserialize;
use std::collections::HashMap;

#[derive(Parser, Debug)]
#[command(about = "Analyze ORE game history")]
struct Args {
    /// RPC URL
    #[arg(short, long)]
    rpc: String,

    /// Number of recent rounds to analyze
    #[arg(short, long)]
    rounds: Option<u64>,

    /// Start round ID
    #[arg(short, long)]
    start: Option<u64>,

    /// End round ID
    #[arg(short, long)]
    end: Option<u64>,

    /// Output CSV file
    #[arg(short, long, default_value = "rounds.csv")]
    csv: String,

    /// Output JSON file
    #[arg(short, long, default_value = "rounds.json")]
    json: String,

    /// Cache file for storing fetched rounds
    #[arg(long, default_value = "rounds_cache.json")]
    cache: String,

    /// Skip loading from cache
    #[arg(long)]
    no_cache: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RoundData {
    round_id: u64,
    winning_square: Option<u64>,
    total_deployed: f64,
    total_winnings: f64,
    total_vaulted: f64,
    num_winners: u64,
    top_miner_reward: f64,
    motherlode: f64,
    was_split: bool,
    was_void: bool,
    deployments: Vec<f64>,
}

fn lamports_to_sol(lamports: u64) -> f64 {
    lamports as f64 / 1_000_000_000.0
}

fn ore_to_tokens(ore: u64) -> f64 {
    ore as f64 / 100_000_000_000.0
}

async fn get_board(rpc: &RpcClient) -> Result<Board> {
    let board_pda = ore_api::state::board_pda();
    let account = rpc.get_account(&board_pda.0).await?;
    let board = Board::try_from_bytes(&account.data)?;
    Ok(*board)
}

async fn get_round(rpc: &RpcClient, id: u64) -> Result<Option<Round>> {
    let round_pda = ore_api::state::round_pda(id);
    let account = match rpc.get_account(&round_pda.0).await {
        Ok(acc) => acc,
        Err(_) => return Ok(None),
    };
    let round = Round::try_from_bytes(&account.data)?;
    Ok(Some(*round))
}

async fn fetch_round_data(rpc: &RpcClient, round_id: u64) -> Result<Option<RoundData>> {
    let Some(round) = get_round(rpc, round_id).await? else {
        return Ok(None);
    };

    // Determine winning square
    let winning_square = if let Some(rng) = round.rng() {
        Some(round.winning_square(rng) as u64)
    } else {
        None
    };

    // Check if void round
    let was_void = round.slot_hash == [0; 32] || round.slot_hash == [u8::MAX; 32];

    // Check if split
    let was_split = if let Some(rng) = round.rng() {
        round.is_split_reward(rng)
    } else {
        false
    };

    // Convert deployments to SOL
    let deployments: Vec<f64> = round.deployed.iter().map(|&d| lamports_to_sol(d)).collect();

    Ok(Some(RoundData {
        round_id,
        winning_square,
        total_deployed: lamports_to_sol(round.total_deployed),
        total_winnings: lamports_to_sol(round.total_winnings),
        total_vaulted: lamports_to_sol(round.total_vaulted),
        num_winners: round.count.get(winning_square.unwrap_or(0) as usize).copied().unwrap_or(0),
        top_miner_reward: ore_to_tokens(round.top_miner_reward),
        motherlode: ore_to_tokens(round.motherlode),
        was_split,
        was_void,
        deployments,
    }))
}

#[derive(Debug)]
struct Statistics {
    total_rounds: usize,
    void_rounds: usize,
    split_rounds: usize,
    motherlode_hits: usize,
    winning_square_freq: HashMap<u64, usize>,
    avg_total_deployed: f64,
    avg_prize_pool: f64,
    avg_per_square: Vec<f64>,
}

fn calculate_statistics(rounds: &[RoundData]) -> Statistics {
    let mut stats = Statistics {
        total_rounds: rounds.len(),
        void_rounds: 0,
        split_rounds: 0,
        motherlode_hits: 0,
        winning_square_freq: HashMap::new(),
        avg_total_deployed: 0.0,
        avg_prize_pool: 0.0,
        avg_per_square: vec![0.0; 25],
    };

    let mut total_deployed_sum = 0.0;
    let mut total_winnings_sum = 0.0;

    for round in rounds {
        // Count void rounds
        if round.was_void {
            stats.void_rounds += 1;
        }

        // Count split rounds
        if round.was_split {
            stats.split_rounds += 1;
        }

        // Count motherlode hits
        if round.motherlode > 0.0 {
            stats.motherlode_hits += 1;
        }

        // Track winning square frequency
        if let Some(square) = round.winning_square {
            *stats.winning_square_freq.entry(square).or_insert(0) += 1;
        }

        // Sum for averages
        total_deployed_sum += round.total_deployed;
        total_winnings_sum += round.total_winnings;

        // Sum per-square deployments
        for (i, &deployed) in round.deployments.iter().enumerate() {
            stats.avg_per_square[i] += deployed;
        }
    }

    // Calculate averages
    let n = rounds.len() as f64;
    stats.avg_total_deployed = total_deployed_sum / n;
    stats.avg_prize_pool = total_winnings_sum / n;
    
    for avg in &mut stats.avg_per_square {
        *avg /= n;
    }

    stats
}

fn print_statistics(stats: &Statistics) {
    println!("\n=== ORE Game Historical Analysis ===\n");
    
    println!("Total Rounds Analyzed: {}", stats.total_rounds);
    println!("Void Rounds: {} ({:.2}%)", stats.void_rounds, 
             stats.void_rounds as f64 / stats.total_rounds as f64 * 100.0);
    println!("Split Reward Rounds: {} ({:.2}%)", stats.split_rounds,
             stats.split_rounds as f64 / stats.total_rounds as f64 * 100.0);
    println!("Motherlode Hits: {} ({:.2}%)", stats.motherlode_hits,
             stats.motherlode_hits as f64 / stats.total_rounds as f64 * 100.0);
    
    println!("\nAverage Metrics:");
    println!("  Total Deployed per Round: {:.4} SOL", stats.avg_total_deployed);
    println!("  Prize Pool per Round: {:.4} SOL", stats.avg_prize_pool);
    
    println!("\nWinning Square Frequency:");
    let mut freq_vec: Vec<_> = stats.winning_square_freq.iter().collect();
    freq_vec.sort_by_key(|&(square, _)| square);
    for (square, count) in freq_vec {
        let pct = *count as f64 / (stats.total_rounds - stats.void_rounds) as f64 * 100.0;
        println!("  Square #{:2}: {} wins ({:.2}%)", square, count, pct);
    }
    
    println!("\nAverage Deployment per Square:");
    for (i, &avg) in stats.avg_per_square.iter().enumerate() {
        if i % 5 == 0 {
            println!();
        }
        print!("  #{:2}: {:.4} SOL  ", i, avg);
    }
    println!("\n");
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    let rpc = RpcClient::new(args.rpc);
    
    // Determine round range
    let (start_round, end_round) = if let Some(n) = args.rounds {
        let board = get_board(&rpc).await?;
        let current = board.round_id;
        (current.saturating_sub(n), current)
    } else if let (Some(start), Some(end)) = (args.start, args.end) {
        (start, end)
    } else {
        eprintln!("Must specify either --rounds or both --start and --end");
        std::process::exit(1);
    };
    
    // Load cache if it exists and not disabled
    let mut cache: HashMap<u64, RoundData> = if !args.no_cache && std::path::Path::new(&args.cache).exists() {
        println!("Loading cache from {}...", args.cache);
        let cache_data = std::fs::read_to_string(&args.cache)?;
        let cached_rounds: Vec<RoundData> = serde_json::from_str(&cache_data)?;
        let cache_map: HashMap<u64, RoundData> = cached_rounds.into_iter()
            .map(|r| (r.round_id, r))
            .collect();
        println!("Loaded {} rounds from cache", cache_map.len());
        cache_map
    } else {
        HashMap::new()
    };
    
    println!("Fetching rounds {} to {}...", start_round, end_round);
    
    // Fetch all rounds (or load from cache)
    let mut rounds = Vec::new();
    let mut fetched = 0;
    let mut from_cache = 0;
    let mut failed = 0;
    
    for round_id in start_round..=end_round {
        // Check cache first
        if let Some(cached_round) = cache.get(&round_id) {
            rounds.push(cached_round.clone());
            from_cache += 1;
            continue;
        }
        
        // Fetch from RPC if not in cache
        match fetch_round_data(&rpc, round_id).await {
            Ok(Some(data)) => {
                cache.insert(round_id, data.clone());
                rounds.push(data);
                fetched += 1;
                if fetched % 10 == 0 {
                    print!(".");
                    if fetched % 100 == 0 {
                        println!(" {} fetched", fetched);
                    }
                }
            }
            Ok(None) => {
                failed += 1;
            }
            Err(e) => {
                if fetched % 10 != 0 {
                    println!();
                }
                eprintln!("Error fetching round {}: {}", round_id, e);
                failed += 1;
            }
        }
    }
    
    println!("\n\nFrom cache: {} rounds", from_cache);
    println!("Fetched from RPC: {} rounds", fetched);
    println!("Failed: {} rounds", failed);
    
    // Save updated cache
    if fetched > 0 {
        println!("Updating cache...");
        let cache_rounds: Vec<RoundData> = cache.values().cloned().collect();
        let cache_json = serde_json::to_string_pretty(&cache_rounds)?;
        std::fs::write(&args.cache, cache_json)?;
        println!("Cache updated: {} total rounds stored", cache.len());
    }
    
    if rounds.is_empty() {
        eprintln!("No rounds fetched. Exiting.");
        return Ok(());
    }
    
    // Calculate statistics
    let stats = calculate_statistics(&rounds);
    print_statistics(&stats);
    
    // Export to CSV
    let mut wtr = csv::Writer::from_path(&args.csv)?;
    for round in &rounds {
        wtr.serialize(round)?;
    }
    wtr.flush()?;
    println!("CSV written to: {}", args.csv);
    
    // Export to JSON
    let json = serde_json::to_string_pretty(&rounds)?;
    std::fs::write(&args.json, json)?;
    println!("JSON written to: {}", args.json);
    
    Ok(())
}