import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.*;
import java.util.Arrays;
import java.util.List;

public class MyWord2Vector {
    public static void main(String[] args){
        SparkConf conf = new SparkConf().setAppName("Word2Vector").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SQLContext sqlContext = new SQLContext(sc);

        // Input data: Each row is a bag of words from a sentence or document.
        List<Row> data = Arrays.asList(
                RowFactory.create(Arrays.asList("Hi I heard about Spark".split(" "))),
                RowFactory.create(Arrays.asList("I wish Java could use case classes".split(" "))),
                RowFactory.create(Arrays.asList("Logistic regression models are neat".split(" ")))
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("text", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
        });
        Dataset<Row> documentDF = sqlContext.createDataFrame(data, schema);

        // Learn a mapping from words to Vectors.
        Word2Vec word2Vec = new Word2Vec()
                .setInputCol("text")
                .setOutputCol("result")
                .setVectorSize(3)
                .setMinCount(0);
        Word2VecModel model = word2Vec.fit(documentDF);
        Dataset<Row> result = model.transform(documentDF);

        for(Row  row : result.collectAsList()){
            List<String> text  = row.getList(0);
            Vector vector = (Vector)row.get(1);
            System.out.println("Text: " + text + "\t=>\t Vector: " + vector);
        }

//Text: [Hi, I, heard, about, Spark]	=>	 Vector: [-0.02205655723810196]
//Text: [I, wish, Java, could, use, case, classes]	=>	 Vector: [-0.009554644780499595]
//Text: [Logistic, regression, models, are, neat]	=>	 Vector: [-0.12159877410158515]

        sc.stop();
    }
}