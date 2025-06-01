from src.beyblade_battle_analyzer import logger
from src.beyblade_battle_analyzer.entity.config_entity import BattleSummaryConfig


class BattleSummary:
    """
    Class to generate a summary of the Beyblade battle.
    """

    def __init__(self, config: BattleSummaryConfig, result):
        self.config = config
        self.battle_summary = result['battle_summary']

    def generate_summary(self):
        """
        Generates and log the battle summary.
        """
        logger.info("Generating battle summary...")

        if self.battle_summary.get('battle_duration'):
            duration = self.battle_summary['battle_duration']
            logger.info(f"⏱️  Battle Duration: {duration:.1f} seconds")

        winner_analysis = self.battle_summary.get('winner_analysis', {})
        if self.battle_summary.get('winner_id') is not None:
            winner_id = self.battle_summary['winner_id']
            logger.info(f'🏆 Winner: Beyblade #{winner_id}')
            logger.info('🎯 Results: ' + winner_analysis.get('reason', 'Battle completed'))

        elif winner_analysis.get('current_leader_id') is not None:
            leader_id = winner_analysis['current_leader_id']
            active_count = len(winner_analysis.get('active_beyblades', []))
            logger.info("🔄 Battle Status: ONGOING")
            logger.info(f'🥇 Current Leader: Beyblade #{leader_id}')
            logger.info(f'🎮 Active Beyblades: {active_count}')
        else:
            logger.info("🔄 Battle Status: " + self.battle_summary.get('battle_status', 'unknown').upper())
            logger.info("🎮 No Beyblades are currently active.")
