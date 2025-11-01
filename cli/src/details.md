ORE Game Mechanics Breakdown:

 Basic Structure
  - 5x5 grid (25 squares total)
  - Rounds last 150 slots (approximately 60 seconds)
  - 35 slot intermission between rounds
  - Max supply: 5 million ORE tokens

 How to Play
  - Players "deploy" SOL on one or more squares
  - Each square tracks total SOL deployed and number of miners
  - Players can deploy to multiple squares per round
  - Must checkpoint previous round before deploying in new round

 Round Resolution (Reset)
  - At round end (150 slots), winning square chosen via slot hash (cryptographically random)
  - Slot hash must be retrieved within ~2.5 minutes or round is voided (all SOL refunded)
  - Winning square determined: rng % 25

 Reward Distribution
  - Winners: All players who deployed on winning square
  - Prize pool: Total SOL from all non-winning squares
  - 1% admin fee taken from total deployed
  - 10% of winnings goes to treasury vault
  - Remaining 89% split proportionally among winners based on deployment amount

 ORE Token Rewards
  - +1 ORE minted per round (split among winning miners)
  - 25% chance the +1 ORE is split equally among ALL winners
  - +0.2 ORE minted to motherlode pool every round
  - 1/625 chance (0.16%) to hit motherlode and win entire pool

 Checkpointing
  - Players must checkpoint after each round to claim rewards
  - Checkpoint fee: 0.00001 SOL (paid once, held in miner account)
  - Rewards accumulate until claimed

 EV Calculation
  - Your bot calculates expected value based on:
    - Current deployments per square
    - ORE/SOL price
    - Motherlode size
    - Protocol fees (1% admin + 10% vault)
  - Formula optimizes bet size to maximize expected return</tool_message>