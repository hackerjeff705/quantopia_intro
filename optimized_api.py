import pandas as pd

import quantopian.algorithm as algo
import quantopian.experimental.optimize as opt

from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import builtin, morningstar as mstar
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.factors.morningstar import MarketCap
from quantopian.pipeline.classifiers.morningstar import Sector

from quantopian.pipeline.filters.morningstar import Q1500US
from quantopian.pipeline.data.sentdex import sentiment
from quantopian.pipeline.data.morningstar import operation_ratios

# Algorithm Parameters
# --------------------
# Universe Selection Parameters
UNIVERSE_SIZE = 500
MIN_MARKET_CAP_PERCENTILE = 50
LIQUIDITY_LOOKBACK_LENGTH = 100

# Constraint Parameters
MAX_GROSS_LEVERAGE = 1.0
MAX_SHORT_POSITION_SIZE = 0.015  # 1.5%
MAX_LONG_POSITION_SIZE = 0.015  # 1.5%

# Scheduling Parameters
MINUTES_AFTER_OPEN_TO_TRADE = 10
BASE_UNIVERSE_RECALCULATE_FREQUENCY = 'month_start'  # {week,quarter,year}_start are also valid


def initialize(context):
    # try many types i.e.
    # Yes: operation_ratios.revenue_growth.latest, operation_ratios.operation_margin, sentiment
    testing_factor1 = operation_ratios.operation_margin.latest
    testing_factor2 = operation_ratios.revenue_growth.latest
    testing_factor3 = sentiment.sentiment_signal.latest

    universe = (Q1500US() &
                testing_factor1.notnull() &
                testing_factor2.notnull() &
                testing_factor3.notnull())

    testing_factor1 = testing_factor1.rank(mask=universe, method='average')
    testing_factor2 = testing_factor2.rank(mask=universe, method='average')
    testing_factor3 = testing_factor3.rank(mask=universe, method='average')

    combined_alpha = testing_factor1 + testing_factor2 + testing_factor3

    # Schedule Tasks
    # --------------
    # Create and register a pipeline computing our combined alpha and a sector
    # code for every stock in our universe. We'll use these values in our
    # optimization below.
    pipe = Pipeline(
        columns={
            'alpha': combined_alpha,
            'sector': Sector(),
        },
        # combined_alpha will be NaN for all stocks not in our universe,
        # but we also want to make sure that we have a sector code for everything
        # we trade.
        screen=combined_alpha.notnull() & Sector().notnull(),
    )
    algo.attach_pipeline(pipe, 'pipe')

    # Schedule a function, 'do_portfolio_construction', to run once a week
    # ten minutes after market open.
    algo.schedule_function(
        do_portfolio_construction,
        date_rule=algo.date_rules.week_start(),
        time_rule=algo.time_rules.market_open(minutes=MINUTES_AFTER_OPEN_TO_TRADE),
        half_days=False,
    )


def before_trading_start(context, data):
    # Call pipeline_output in before_trading_start so that pipeline
    # computations happen in the 5 minute timeout of BTS instead of the 1
    # minute timeout of handle_data/scheduled functions.
    context.pipeline_data = algo.pipeline_output('pipe')


# Portfolio Construction
# ----------------------
def do_portfolio_construction(context, data):
    pipeline_data = context.pipeline_data
    todays_universe = pipeline_data.index

    # Objective
    # ---------
    # For our objective, we simply use our naive ranks as an alpha coefficient
    # and try to maximize that alpha.
    #
    # This is a **very** naive model. Since our alphas are so widely spread out,
    # we should expect to always allocate the maximum amount of long/short
    # capital to assets with high/low ranks.
    #
    # A more sophisticated model would apply some re-scaling here to try to generate
    # more meaningful predictions of future returns.
    objective = opt.MaximizeAlpha(pipeline_data.alpha)

    # Constraints
    # -----------
    # Constrain our gross leverage to 1.0 or less. This means that the absolute
    # value of our long and short positions should not exceed the value of our
    # portfolio.
    constrain_gross_leverage = opt.MaxGrossLeverage(MAX_GROSS_LEVERAGE)

    # Constrain individual position size to no more than a fixed percentage
    # of our portfolio. Because our alphas are so widely distributed, we
    # should expect to end up hitting this max for every stock in our universe.
    constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
        -MAX_SHORT_POSITION_SIZE,
        MAX_LONG_POSITION_SIZE,
    )

    # Constrain ourselves to allocate the same amount of capital to
    # long and short positions.
    market_neutral = opt.DollarNeutral()

    # Constrain ourselve to have a net leverage of 0.0 in each sector.
    sector_neutral = opt.NetPartitionExposure.with_equal_bounds(
        labels=pipeline_data.sector,
        min=-0.0001,
        max=0.0001,
    )

    # Run the optimization. This will calculate new portfolio weights and
    # manage moving our portfolio toward the target.
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=[
            constrain_gross_leverage,
            constrain_pos_size,
            market_neutral,
            sector_neutral,
        ],
        universe=todays_universe,
    )