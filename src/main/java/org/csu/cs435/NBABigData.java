package org.csu.cs435;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF9;
import java.util.*;
import static org.apache.spark.sql.functions.*;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

public class NBABigData {

    // Mapping of event types to eWPA values
    public static Map<String, Double> eWPAValues = new HashMap<>();

    static {
        eWPAValues.put("Assist", 0.012);
        eWPAValues.put("Made 3-Point Shot", 0.046);
        eWPAValues.put("Missed 3-Point Shot", -0.03);
        eWPAValues.put("Made 2-Point Shot", 0.022);
        eWPAValues.put("Missed 2-Point Shot", -0.014);
        eWPAValues.put("Made Free Throw", 0.005);
        eWPAValues.put("Missed Free Throw", -0.015);
        eWPAValues.put("Rebound", 0.002); //I have failed to find a reliable way to determine if a rebound is O or D, so I just made rebound the average of the two. should barely make a difference I think
        eWPAValues.put("Turnover", -0.021);
        eWPAValues.put("STEAL" , .021);
        eWPAValues.put("BLOCK" , .011); //remember to adjust this
        // Add under 10 seconds weights
        eWPAValues.put("Assist (Last 10 Seconds)", 0.018);
        eWPAValues.put("Made 3-Point Shot (Last 10 Seconds)", 0.069);
        eWPAValues.put("Made 2-Point Shot (Last 10 Seconds)", 0.033);
        eWPAValues.put("Made Free Throw (Last 10 Seconds)", 0.010);
        eWPAValues.put("Missed Free Throw (Last 10 Seconds)", -0.030);
        eWPAValues.put("Rebound (Last 10 Seconds)", 0.004);
        eWPAValues.put("Turnover (Last 10 Seconds)", -0.042);
        eWPAValues.put("STEAL (Last 10 Seconds)" , .042);
        eWPAValues.put("BLOCK (Last 10 Seconds)" , .022);
        eWPAValues.put("Made 3-Point Shot (Clutch Margin)", 0.1);
        eWPAValues.put("Missed 3-Point Shot (Clutch Margin)", -0.1);
        eWPAValues.put("Made 2-Point Shot (Clutch Margin)", 0.060);
        eWPAValues.put("Missed 2-point Shot (Clutch Margin)", -0.060);
    }


    public static void main(String[] args) {

        SparkSession spark = createSparkSession();

        Dataset<Row> nbaPlayByPlay = readData(spark, args[0]);

        // Preprocessing Steps
        Dataset<Row> preprocessedData = preprocessData(nbaPlayByPlay, spark);

        // Proceed with Clutchness Calculation
        Dataset<Row> clutchScores = calculateClutchness(preprocessedData, spark);

        // Show the top 10 players by Adjusted Clutchness Score
        System.out.println("Top 10 Players by Adjusted Clutchness Score:");
        clutchScores.orderBy(functions.desc("Adjusted_eWPA")).show(10);

        // Corrected grouping by PLAYER_ID and PLAYER_NAME
        Dataset<Row> playerTotalClutchScores = clutchScores.groupBy("PLAYER_ID", "PLAYER_NAME")
                .agg(functions.sum("Adjusted_eWPA").alias("Total_ClutchScore"));

        playerTotalClutchScores = playerTotalClutchScores.filter(col("PLAYER_NAME").isNotNull());
        // This aggregates all the seasons together for each player
        // Show the top 10 players by total clutch score
        System.out.println("Top 10 Players overall season Adjusted Clutchness Score:");
        playerTotalClutchScores.orderBy(functions.desc("Total_ClutchScore")).show(100);

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
        spark.udf().register("timeStringToSeconds", (UDF2<String, String, Integer>) NBABigData::convertTimeStringToSeconds, DataTypes.IntegerType);

    // Extract hours and minutes from PCTIMESTRING
        df = df.withColumn("seconds_part", functions.minute(df.col("PCTIMESTRING")).cast(DataTypes.StringType));
        df = df.withColumn("minutes_part", functions.hour(df.col("PCTIMESTRING")).cast(DataTypes.StringType));

    // PCTIMESTRING is the time on the clock as a timestampobject
        df = df.withColumn("SECONDS_REMAINING", functions.callUDF("timeStringToSeconds", df.col("minutes_part"), df.col("seconds_part")));

        // Replace "TIE" with 0 directly in the SCOREMARGIN column
        df = df.withColumn("SCOREMARGIN",
                when(col("SCOREMARGIN").equalTo("TIE"), lit(0))  // If SCOREMARGIN is "TIE", set it to 0
                        .otherwise(col("SCOREMARGIN")) // Otherwise, keep the original value
        );

        // Convert SCOREMARGIN to Integer type
        df = df.withColumn("SCOREMARGIN", col("SCOREMARGIN").cast(DataTypes.IntegerType));

        // We need to partition by game ID and order by event num so we can forward propogate the stupid scoremargin
        WindowSpec windowSpec = Window.partitionBy("GAME_ID")
                .orderBy("EVENTNUM")
                .rowsBetween(Window.unboundedPreceding(), Window.currentRow());

        // Fill forward the last non-null SCOREMARGIN
        df = df.withColumn("SCOREMARGIN_FFILLED", last(col("SCOREMARGIN"), true).over(windowSpec));

        // Filter events where the score margin is within 6 points
        df = df.filter(abs(col("SCOREMARGIN_FFILLED")).leq(6));

        //replace SCOREMARGIN with the filled values and get rid of forward filled
        df = df.withColumn("SCOREMARGIN", col("SCOREMARGIN_FFILLED")).drop("SCOREMARGIN_FFILLED");

        // we only want the last 5 minutes
        df= df.filter(
                functions.col("SECONDS_REMAINING").isNotNull()
                        .and(functions.col("SECONDS_REMAINING").geq(0))
                        .and(functions.col("SECONDS_REMAINING").leq(300))
        );

        return df;
    }

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

    public static int convertTimeStringToSeconds(String minutes, String seconds) {
        if (minutes == null || minutes.isEmpty() || seconds == null || seconds.isEmpty()) {
            return -1;
        }
        try {
            int minutesInt = Integer.parseInt(minutes);
            int secondsInt = Integer.parseInt(seconds);
            return minutesInt * 60 + secondsInt;
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    public static List<String> classifyEvent(Integer eventMsgType, Integer eventMsgActionType, String homeDescription,
                                             String awayDescription, Integer secondsRemaining, Integer scoreMargin,
                                             String player1Name, String player2Name, String player3Name) {
        List<String> eventTypes = new ArrayList<>();

        if (eventMsgType == null) {
            eventTypes.add("Other");
            return eventTypes;
        }

        boolean isLast10Seconds = secondsRemaining != null && (secondsRemaining <= 10) && secondsRemaining > 0;

        int absScoreMargin = Math.abs(scoreMargin);

        // Normalize descriptions to uppercase for case-insensitive matching
        if (homeDescription != null) homeDescription = homeDescription.toUpperCase();
        if (awayDescription != null) awayDescription = awayDescription.toUpperCase();

        // Detect assists
        if (isAssist(homeDescription, awayDescription)) {
            eventTypes.add(isLast10Seconds ? "Assist (Last 10 Seconds)" : "Assist");
        }

        // Detect steals
        if (isSteal(homeDescription, awayDescription)) {
            eventTypes.add(isLast10Seconds ? "STEAL (Last 10 Seconds)" : "STEAL");
        }

        // Detect blocks
        if (isBlock(homeDescription, awayDescription)) {
            eventTypes.add(isLast10Seconds ? "BLOCK (Last 10 Seconds)" : "BLOCK");
        }

        // Detect turnovers
        if (isTurnover(homeDescription, awayDescription)) {
            if (player1Name != null || player2Name != null || player3Name != null) {
                eventTypes.add(isLast10Seconds ? "Turnover (Last 10 Seconds)" : "Turnover");
            }
        }

        // Detect rebounds
        if (isRebound(homeDescription, awayDescription)) {
            if (player1Name != null || player2Name != null || player3Name != null) {
                eventTypes.add(isLast10Seconds ? "Rebound (Last 10 Seconds)" : "Rebound");
            }
        }

        // Detect shots (3-pointers and 2-pointers, made and missed)
        if (eventMsgActionType != null) {
            if (isThreePointer(homeDescription, awayDescription)) {
                if (isMissedShot(homeDescription, awayDescription)) {
                    eventTypes.add((isLast10Seconds && absScoreMargin <= 3) ? "Missed 3-Point Shot (Clutch Margin)" : "Missed 3-Point Shot");
                } else {
                    if (isLast10Seconds && homeDescription != null && scoreMargin <= 0 && absScoreMargin <= 3) {
                        eventTypes.add("Made 3-Point Shot (Clutch Margin)");
                    } else if(isLast10Seconds && homeDescription == null && scoreMargin >= 0 && absScoreMargin <= 3){
                        eventTypes.add("Made 3-Point Shot (Clutch Margin)");
                    }else {
                        eventTypes.add(isLast10Seconds ? "Made 3-Point Shot (Last 10 Seconds)" : "Made 3-Point Shot");
                    }
                }
            } else if (isTwoPointer(homeDescription, awayDescription)) {
                if (isMissedShot(homeDescription, awayDescription)) {
                    eventTypes.add((isLast10Seconds && absScoreMargin <= 2) ? "Missed 2-Point Shot (Clutch Margin)" : "Missed 2-Point Shot");
                } else {
                    if ((isLast10Seconds && homeDescription != null && scoreMargin <= 0 && absScoreMargin <= 2) || isLast10Seconds && homeDescription == null && scoreMargin >= 0 && absScoreMargin <= 2) {
                        eventTypes.add("Made 2-Point Shot (Clutch Margin)");
                    } else {
                        eventTypes.add(isLast10Seconds ? "Made 2-Point Shot (Last 10 Seconds)" : "Made 2-Point Shot");
                    }
                }
            }
        }

        // Detect free throws
        if (isFreeThrow(homeDescription, awayDescription)) {
            if (isMissedShot(homeDescription, awayDescription)) {
                eventTypes.add(isLast10Seconds ? "Missed Free Throw (Last 10 Seconds)" : "Missed Free Throw");
            } else {
                eventTypes.add(isLast10Seconds ? "Made Free Throw (Last 10 Seconds)" : "Made Free Throw");
            }
        }

        if (eventTypes.isEmpty()) {
            eventTypes.add("Other");
        }

        return eventTypes;
    }


//helpers, now with all caps
    private static boolean isThreePointer(String homeDescription, String visitorDescription) {
        return (homeDescription != null && homeDescription.contains("3PT")) ||
                (visitorDescription != null && visitorDescription.contains("3PT"));
    }

    private static boolean isTwoPointer(String homeDescription, String visitorDescription) {
        return ((homeDescription != null && !homeDescription.contains("3PT") &&
                (homeDescription.contains("DUNK") || homeDescription.contains("LAYUP") || homeDescription.contains("SHOT"))) ||
                (visitorDescription != null && !visitorDescription.contains("3PT") &&
                        (visitorDescription.contains("DUNK") || visitorDescription.contains("LAYUP") || visitorDescription.contains("SHOT"))));
    }

    private static boolean isMissedShot(String homeDescription, String visitorDescription) {
        return (homeDescription != null && homeDescription.contains("MISS")) ||
                (visitorDescription != null && visitorDescription.contains("MISS"));
    }

    private static boolean isFreeThrow(String homeDescription, String visitorDescription) {
        return (homeDescription != null && homeDescription.contains("FREE THROW")) ||
                (visitorDescription != null && visitorDescription.contains("FREE THROW"));
    }

    private static boolean isRebound(String homeDescription, String visitorDescription) {
        return (homeDescription != null && homeDescription.contains("REBOUND")) ||
                (visitorDescription != null && visitorDescription.contains("REBOUND"));
    }

    public static Boolean isTurnover(String homeDescription, String visitorDescription) {
        return (homeDescription != null && homeDescription.contains("TURNOVER")) ||
                (visitorDescription != null && visitorDescription.contains("TURNOVER"));
    }

    public static Boolean isAssist(String homeDescription, String visitorDescription) {
        return (homeDescription != null && homeDescription.contains("AST")) ||
                (visitorDescription != null && visitorDescription.contains("AST"));
    }

    public static Boolean isSteal(String homeDescription, String visitorDescription) {
        return (homeDescription != null && homeDescription.contains("STEAL")) ||
                (visitorDescription != null && visitorDescription.contains("STEAL"));
    }

    public static Boolean isBlock(String homeDescription, String visitorDescription) {
        return (homeDescription != null && homeDescription.contains("BLOCK")) ||
                (visitorDescription != null && visitorDescription.contains("BLOCK"));
    }



    public static double getEwpaValue(String eventType) {
        return eWPAValues.getOrDefault(eventType, 0.0);
    }

    private static Dataset<Row> calculateClutchness(Dataset<Row> df, SparkSession spark) {
        // Prepare descriptions
        df = df.withColumn("HOMEDESCRIPTION", functions.coalesce(df.col("HOMEDESCRIPTION"), functions.lit("")))
                .withColumn("VISITORDESCRIPTION", functions.coalesce(df.col("VISITORDESCRIPTION"), functions.lit("")));

        // Register UDFs
        spark.udf().register("classifyEvent",
                (UDF9<Integer, Integer, String, String, Integer, Integer, String, String, String, List<String>>) NBABigData::classifyEvent,
                DataTypes.createArrayType(DataTypes.StringType));
        spark.udf().register("getEwpaValue", (UDF1<String, Double>) NBABigData::getEwpaValue, DataTypes.DoubleType);

        // Classify events (returns an array of event types)
        df = df.withColumn("EVENT_TYPES", functions.callUDF("classifyEvent",
                df.col("EVENTMSGTYPE"), df.col("EVENTMSGACTIONTYPE"), df.col("HOMEDESCRIPTION"),
                df.col("VISITORDESCRIPTION"), df.col("SECONDS_REMAINING"), df.col("SCOREMARGIN"),
                df.col("PLAYER1_NAME"), df.col("PLAYER2_NAME"), df.col("PLAYER3_NAME")));

        // Explode EVENT_TYPES to create one row per event type
        df = df.withColumn("EVENT_TYPE", functions.explode(df.col("EVENT_TYPES")));

        // Assign eWPA values for each event type
        df = df.withColumn("eWPA", functions.callUDF("getEwpaValue", df.col("EVENT_TYPE")));

        // Initialize eWPA columns for each player
        df = df.withColumn("PLAYER1_eWPA", functions.lit(0.0))
                .withColumn("PLAYER2_eWPA", functions.lit(0.0))
                .withColumn("PLAYER3_eWPA", functions.lit(0.0));

        // Assign eWPA to players based on event type and role
        df = assignEwpaToPlayers(df);

        // Create an array of structs containing PLAYER_ID, PLAYER_NAME, and eWPA
        df = df.withColumn("players_eWPA", functions.array(
                functions.struct(df.col("PLAYER1_ID").alias("PLAYER_ID"), df.col("PLAYER1_NAME").alias("PLAYER_NAME"), df.col("PLAYER1_eWPA").alias("eWPA")),
                functions.struct(df.col("PLAYER2_ID").alias("PLAYER_ID"), df.col("PLAYER2_NAME").alias("PLAYER_NAME"), df.col("PLAYER2_eWPA").alias("eWPA")),
                functions.struct(df.col("PLAYER3_ID").alias("PLAYER_ID"), df.col("PLAYER3_NAME").alias("PLAYER_NAME"), df.col("PLAYER3_eWPA").alias("eWPA"))
        ));

        // Explode the array to create one row per player per event
        df = df.withColumn("player_eWPA", functions.explode(df.col("players_eWPA")));

        Dataset<Row> playerData = df.select(
                "SEASON_TYPE",
                "player_eWPA.PLAYER_ID",
                "player_eWPA.PLAYER_NAME",
                "player_eWPA.eWPA",
                "EVENT_TYPE",
                "HOMEDESCRIPTION",
                "VISITORDESCRIPTION",
                "SECONDS_REMAINING",
                "SCOREMARGIN"
        );
        String playerName = "John Stockton"; //change this line for the player you want to see
        Dataset<Row> playerEvents = playerData.filter(col("PLAYER_NAME").equalTo(playerName));
        playerEvents.show(100, false);
        // Select the relevant columns
        df = df.select("SEASON_TYPE", "player_eWPA.PLAYER_ID", "player_eWPA.PLAYER_NAME", "player_eWPA.eWPA");

        // Filter out null PLAYER_IDs and zero eWPA
        df = df.filter(df.col("PLAYER_ID").isNotNull().and(df.col("eWPA").notEqual(0.0)));

        // Aggregate clutchness scores per player and season type
        Dataset<Row> playerClutchScores = df.groupBy("PLAYER_ID", "PLAYER_NAME", "SEASON_TYPE")
                .agg(functions.sum("eWPA").alias("Total_eWPA"));

        // Adjust for game importance FiX tHiS laTER
       /* playerClutchScores = playerClutchScores.withColumn("Adjusted_eWPA", functions.when(
                functions.col("SEASON_TYPE").equalTo("Regular Season"), functions.col("Total_eWPA")
        ).when(
                functions.col("SEASON_TYPE").equalTo("Playoffs"), functions.col("Total_eWPA").multiply(1.5)
        ).when(
                functions.col("SEASON_TYPE").equalTo("Finals"), functions.col("Total_eWPA").multiply(2.0)
        ).otherwise(functions.col("Total_eWPA")));

        return playerClutchScores;
    }
*/
    private static Dataset<Row> assignEwpaToPlayers(Dataset<Row> df) {
        // Assign eWPA to PLAYER1 for shots, free throws, rebounds, and turnovers
        df = df.withColumn("PLAYER1_eWPA", functions.when(
                functions.col("EVENT_TYPE").rlike("^Made.*Shot.*|^Missed.*Shot.*|^Made Free Throw.*|^Missed Free Throw.*|^Rebound.*|^Turnover.*"),
                df.col("eWPA")
        ).otherwise(df.col("PLAYER1_eWPA")));

        // Assign eWPA to PLAYER2 for assists and steals
        df = df.withColumn("PLAYER2_eWPA", functions.when(
                functions.col("EVENT_TYPE").rlike("^Assist.*|^STEAL.*"),
                df.col("eWPA")
        ).otherwise(df.col("PLAYER2_eWPA")));

        // Assign eWPA to PLAYER3 for blocks
        df = df.withColumn("PLAYER3_eWPA", functions.when(
                functions.col("EVENT_TYPE").rlike("^BLOCK.*"),
                df.col("eWPA")
        ).otherwise(df.col("PLAYER3_eWPA")));

        return df;
    }


}
