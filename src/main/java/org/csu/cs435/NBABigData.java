package org.csu.cs435;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF6;

import java.util.Map;
import java.util.HashMap;

public class NBABigData {

    // Mapping of event types to eWPA values
    public static Map<String, Double> eWPAValues = new HashMap<>();

    static {
        eWPAValues.put("Made 3-Point Shot", 0.040);
        eWPAValues.put("Made 2-Point Shot", 0.020);
        eWPAValues.put("Offensive Rebound", 0.016);
        eWPAValues.put("Getting 1 Foul Shot", 0.016);
        eWPAValues.put("Getting 2 Foul Shots", 0.010);
        eWPAValues.put("Getting 3 Foul Shots", 0.027);
        eWPAValues.put("Defensive Rebound", 0.007);
        eWPAValues.put("Made Free Throw", 0.005);
        eWPAValues.put("Missed Free Throw", -0.015);
        eWPAValues.put("Missed Field Goal", -0.016);
        eWPAValues.put("Turnover", -0.021);
    
        // Add under 10 seconds weights
        eWPAValues.put("Made 3-Point Shot (Last 10 Seconds)", 0.080);
        eWPAValues.put("Made 2-Point Shot (Last 10 Seconds)", 0.040);
        eWPAValues.put("Offensive Rebound (Last 10 Seconds)", 0.032);
        eWPAValues.put("Getting 1 Foul Shot (Last 10 Seconds)", 0.032);
        eWPAValues.put("Getting 2 Foul Shots (Last 10 Seconds)", 0.020);
        eWPAValues.put("Getting 3 Foul Shots (Last 10 Seconds)", 0.054);
        eWPAValues.put("Defensive Rebound (Last 10 Seconds)", 0.014);
        eWPAValues.put("Made Free Throw (Last 10 Seconds)", 0.010);
        eWPAValues.put("Missed Free Throw (Last 10 Seconds)", -0.030);
        eWPAValues.put("Missed Field Goal (Last 10 Seconds)", -0.032);
        eWPAValues.put("Turnover (Last 10 Seconds)", -0.042);

        eWPAValues.put("Made 3-Point Shot (Clutch Margin)", 0.120);
        eWPAValues.put("Made 2-Point Shot (Clutch Margin)", 0.060);
    }
    

    public static void main(String[] args) {

        SparkSession spark = createSparkSession();

        Dataset<Row> nbaPlayByPlay = readData(spark, args[0]);

        // Preprocessing Steps
        Dataset<Row> preprocessedData = preprocessData(nbaPlayByPlay, spark);

        // Show first few rows of the preprocessed data
        System.out.println("Preprocessed Data Sample:");
        preprocessedData.show(5);

        // Proceed with Clutchness Calculation
        Dataset<Row> clutchScores = calculateClutchness(preprocessedData, spark);

        // Show the top 10 players by Adjusted Clutchness Score
        System.out.println("Top 10 Players by Adjusted Clutchness Score:");
        clutchScores.orderBy(functions.desc("Adjusted_eWPA")).show(10);

        // Optionally, save the results to a file
        // clutchScores.write().mode("overwrite").csv("clutch_scores.csv");
    }

    private static SparkSession createSparkSession() {
        return SparkSession.builder()
                .appName("NBA Play-by-Play Data Processing")
                .master("local[*]")
                .getOrCreate();
    }

    private static Dataset<Row> readData(final SparkSession spark, final String filePath) {
        if (filePath.endsWith(".csv")) {
            return spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(filePath);
        } else {
            return spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(filePath + "/*.csv");
        }
    }

    private static Dataset<Row> preprocessData(Dataset<Row> df, SparkSession spark) {
        // Apply Season Type
        df = applySeasonType(df);

        // Keep only 4th quarter and overtime periods, OT is period 5 so this should work
        df = filterByPeriod(df);

        // Register User defined function(UDF) so we can grab time on clock
        spark.udf().register("timeStringToSeconds", (UDF1<String, Integer>) NBABigData::convertTimeStringToSeconds, DataTypes.IntegerType);

        //  PCTIMESTRING is the time on the clock as a String
        df = df.withColumn("SECONDS_REMAINING", functions.callUDF("timeStringToSeconds", df.col("PCTIMESTRING").cast(DataTypes.StringType)));

        // we only want the last 5 minutes
        df = df.filter(functions.col("SECONDS_REMAINING").leq(300));

        // Handle SCOREMARGIN and filter by a diffence of 6
        df = df.withColumn("SCOREMARGIN_INT", functions.when(
                functions.col("SCOREMARGIN").equalTo("TIE"), 0
        ).otherwise(functions.col("SCOREMARGIN").cast(DataTypes.IntegerType)));

        df = df.filter(functions.abs(df.col("SCOREMARGIN_INT")).leq(6));

        // Add a column indicating whether the player's team won (for simplicity, assume home team if PLAYER1 is home)
        df = df.withColumn("TEAM_WIN", functions.when(
                functions.col("SCOREMARGIN_INT").gt(0), functions.lit(true) // Home team wins
        ).otherwise(functions.lit(false))); // Away team wins

        return df;
    }
//I NEED TO TEST IF any of the below IS ACTUALLY BEHAVING CORRECTLY
    private static Dataset<Row> applySeasonType(Dataset<Row> df) {
        // Apply Season Type based on WEEK_OF_SEASON
        df = df.withColumn("SEASON_TYPE", functions.when(
                functions.col("WEEK_OF_SEASON").geq(1).and(functions.col("WEEK_OF_SEASON").leq(24)),
                "Regular Season"
        ).when(
                functions.col("WEEK_OF_SEASON").geq(25).and(functions.col("WEEK_OF_SEASON").leq(30)),
                "Playoffs"
        ).when(
                functions.col("WEEK_OF_SEASON").geq(31).and(functions.col("WEEK_OF_SEASON").leq(33)),
                "Finals"
        ).otherwise("Unknown"));

        return df;
    }

    private static Dataset<Row> filterByPeriod(Dataset<Row> df) {
        // Keep only events from 4th quarter (Period 4) and overtime periods (Period >= 5)
        df = df.filter(functions.col("PERIOD").geq(4));

        return df;
    }

    public static int convertTimeStringToSeconds(String timeString) {
        if (timeString == null || timeString.isEmpty()) {
            return -1;
        }
        String[] parts = timeString.split(":");
        if (parts.length != 2) {
            return -1;
        }
        int minutes = 0;
        int seconds = 0;
        try {
            minutes = Integer.parseInt(parts[0]);
            seconds = Integer.parseInt(parts[1]);
        } catch (NumberFormatException e) {
            // Handle exception if time string is not in expected format, i guess we just can drop the data point
            return -1;
        }
        return minutes * 60 + seconds;
    }
//this all probably needs to be fixed
    public static String classifyEvent(
            Integer eventMsgType, Integer eventMsgActionType, String homeDescription, String awayDescription, Integer secondsRemaining, Integer scoreMargin) {
        
        if (eventMsgType == null) return "Other";

        boolean isLast10Seconds = secondsRemaining <= 10;

        // Detect assists in descriptions (look for key terms like "assist" in descriptions)
        if ((homeDescription != null && homeDescription.contains("AST")) || 
            (awayDescription != null && awayDescription.contains("AST"))) {
            return isLast10Seconds ? "Assist (Last 10 Seconds)" : "Assist";
        }

        // Detect STEAL and BLOCK actions
        if ((homeDescription != null && homeDescription.contains("STEAL")) || 
            (awayDescription != null && awayDescription.contains("STEAL"))) {
            return isLast10Seconds ? "STEAL (Last 10 Seconds)" : "STEAL";
        }

        if ((homeDescription != null && homeDescription.contains("BLOCK")) || 
            (awayDescription != null && awayDescription.contains("BLOCK"))) {
            return isLast10Seconds ? "BLOCK (Last 10 Seconds)" : "BLOCK";
        }

        switch (eventMsgType) {
            case 1: // Made Shot
                if (eventMsgActionType != null && isThreePointer(eventMsgActionType)) {
                    if (isLast10Seconds && 3 > scoreMargin) {
                        return "Made 3-Point Shot (Clutch Margin)";
                    }
                    return isLast10Seconds ? "Made 3-Point Shot (Last 10 Seconds)" : "Made 3-Point Shot";
                } else {
                    if (isLast10Seconds && 2 > scoreMargin) {
                        return "Made 2-Point Shot (Clutch Margin)";
                    }
                    return isLast10Seconds ? "Made 2-Point Shot (Last 10 Seconds)" : "Made 2-Point Shot";
                }
            case 2: // Missed Shot
                if (eventMsgActionType != null && isThreePointer(eventMsgActionType)) {
                    return isLast10Seconds ? "Missed 3-Point Shot (Last 10 Seconds)" : "Missed 3-Point Shot";
                } else {
                    return isLast10Seconds ? "Missed Field Goal (Last 10 Seconds)" : "Missed Field Goal";
                }
            case 3: // Free Throw
                if (eventMsgActionType != null && isMadeFreeThrow(eventMsgActionType)) {
                    return isLast10Seconds ? "Made Free Throw (Last 10 Seconds)" : "Made Free Throw";
                } else {
                    return isLast10Seconds ? "Missed Free Throw (Last 10 Seconds)" : "Missed Free Throw";
                }
            case 4: // Rebound
                return isLast10Seconds ? "Rebound (Last 10 Seconds)" : "Rebound";
            case 5: // Turnover
                return isLast10Seconds ? "Turnover (Last 10 Seconds)" : "Turnover";
            default:
                return "Other";
        }
    }


    private static boolean isThreePointer(int eventMsgActionType) {
        // List of action types corresponding to 3-point shots
        return eventMsgActionType == 79 || eventMsgActionType == 80 || eventMsgActionType == 81 ||
                eventMsgActionType == 82 || eventMsgActionType == 83;
    }

    private static boolean isMadeFreeThrow(int eventMsgActionType) {

        return eventMsgActionType == 10 || eventMsgActionType == 12 || eventMsgActionType == 15;
    }

    public static double getEwpaValue(String eventType) {
        return eWPAValues.getOrDefault(eventType, 0.0);
    }

    private static Dataset<Row> calculateClutchness(Dataset<Row> df, SparkSession spark) {
        df = df.withColumn("HOMEDESCRIPTION", functions.coalesce(df.col("HOMEDESCRIPTION"), functions.lit(""))).withColumn("VISITORDESCRIPTION", functions.coalesce(df.col("VISITORDESCRIPTION"), functions.lit("")));

        // Register UDFs
        spark.udf().register("classifyEvent", (UDF6<Integer, Integer, String, String, Integer, Integer, String>) NBABigData::classifyEvent, DataTypes.StringType);
        spark.udf().register("getEwpaValue", (UDF1<String, Double>) NBABigData::getEwpaValue, DataTypes.DoubleType);

        // Classify events
        df = df.withColumn("EVENT_TYPE", functions.callUDF("classifyEvent", df.col("EVENTMSGTYPE"), df.col("EVENTMSGACTIONTYPE"), df.col("HOMEDESCRIPTION"), df.col("VISITORDESCRIPTION"), df.col("SECONDS_REMAINING"), df.col("SCOREMARGIN_INT")));

        // Assign eWPA values
        df = df.withColumn("eWPA", functions.callUDF("getEwpaValue", df.col("EVENT_TYPE")));

        // Track points, field goals, and free throw attempts
        df = df.withColumn("Points_Scored", functions.when(df.col("EVENT_TYPE").equalTo("Made 3-Point Shot"), 3)
                .when(df.col("EVENT_TYPE").equalTo("Made 2-Point Shot"), 2)
                .when(df.col("EVENT_TYPE").equalTo("Made Free Throw"), 1)
                .otherwise(0));

        df = df.withColumn("Field_Goals_Attempted", functions.when(df.col("EVENT_TYPE").equalTo("Made 3-Point Shot"), 1)
                .when(df.col("EVENT_TYPE").equalTo("Made 2-Point Shot"), 1)
                .when(df.col("EVENT_TYPE").equalTo("Missed 3-Point Shot"), 1)
                .when(df.col("EVENT_TYPE").equalTo("Missed Field Goal"), 1)
                .otherwise(0));

        df = df.withColumn("Free_Throws_Attempted", functions.when(df.col("EVENT_TYPE").equalTo("Getting 1 Foul Shot"), 1)
                .when(df.col("EVENT_TYPE").equalTo("Getting 2 Foul Shots"), 2)
                .when(df.col("EVENT_TYPE").equalTo("Getting 3 Foul Shots"), 3)
                .otherwise(0));

        // Aggregate points scored, FGA, FTA for each player
        Dataset<Row> playerStats = df.groupBy("PLAYER1_ID", "PLAYER1_NAME", "SEASON_TYPE")
                .agg(
                        functions.sum("Points_Scored").alias("Total_Points"),
                        functions.sum("Field_Goals_Attempted").alias("Total_FGA"),
                        functions.sum("Free_Throws_Attempted").alias("Total_FTA"),
                        functions.sum("eWPA").alias("Total_eWPA"), // Keep track of eWPA as before
                        functions.sum(functions.when(functions.col("TEAM_WIN"), 1).otherwise(0)).alias("Total_Wins"),  // Track wins
                        functions.count("TEAM_WIN").alias("Total_Games")  // Track total games played in clutch moments
                );

        // Calculate TS% for each player
        playerStats = playerStats.withColumn("TS_Percentage", functions.when(
                functions.col("Total_FGA").gt(0).and(functions.col("Total_FTA").gt(0)),
                functions.col("Total_Points").divide(functions.lit(2).multiply(functions.col("Total_FGA").plus(functions.lit(0.44).multiply(functions.col("Total_FTA")))))
        ).otherwise(functions.lit(0)));  // If there are no field goal attempts or free throw attempts, TS% is 0

        // Calculate win percentage for each player
        playerStats = playerStats.withColumn("Win_Percentage",
                functions.when(functions.col("Total_Games").gt(0), functions.col("Total_Wins").divide(functions.col("Total_Games"))).otherwise(functions.lit(0)));

        return playerStats;
    }
}
