root = None
workdir = "workdir"
tag = "fmp_economic_exp"
batch_size = 1

downloader = dict(
    type = "FMPEconomicDownloader",
    root = root,
    token = None,
    start_date = "1993-12-01",
    end_date = "2024-01-01",
    interval = "1d",
    delay = 1,
    indicators = [
        "GDP",
        "realGDP",
        "nominalPotentialGDP",
        "realGDPPerCapita",
        "federalFunds",
        "CPI",
        "inflationRate",
        "inflation",
        "retailSales",
        "consumerSentiment",
        "durableGoods",
        "unemploymentRate",
        "totalNonfarmPayroll",
        "initialClaims",
        "industrialProductionTotalIndex",
        "newPrivatelyOwnedHousingUnitsStartedTotalUnits",
        "totalVehicleSales",
        "retailMoneyFunds",
        "smoothedUSRecessionProbabilities",
        "3MonthOr90DayRatesAndYieldsCertificatesOfDeposit",
        "commercialBankInterestRateOnCreditCardPlansAllAccounts",
        "30YearFixedRateMortgageAverage",
        "15YearFixedRateMortgageAverage"
    ],
    stocks_path = "configs/_stock_list_/exp_stocks.txt",
    workdir = workdir,
    tag = tag
)