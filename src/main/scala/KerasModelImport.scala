import scala.util.{Try, Success, Failure}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.modelimport.keras._

import org.datavec.api.records.reader.RecordReader
import org.nd4j.linalg.dataset.DataSet

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation

object KerasModelImportExample {
  def main(args: Array[String]): Unit = {
    val modelJsonPath = "/Users/debasishghosh/projects/dl4j-model-import/src/main/resources/model.json"
    val weightsH5Path = "/Users/debasishghosh/projects/dl4j-model-import/src/main/resources/model.h5"
    val dataFile = "kdd.csv"

    val stats = for {
      model  <- importFromKeras(modelJsonPath, weightsH5Path)
      reader <- getReader(dataFile)
      data   <- readData(reader)
      eval   <- evaluate(model, data)
    } yield eval.stats()

    println(stats)
  }

  private def importFromKeras(jsonPath: String, h5Path: String): Try[MultiLayerNetwork] = Try {
    KerasModelImport.importKerasSequentialModelAndWeights(jsonPath, h5Path, false)
  }

  private def getReader(dataFile: String): Try[RecordReader] = Try {
    val numLinesToSkip = 0
    val delimiter = ","
    val reader = new CSVRecordReader(numLinesToSkip, delimiter)
    reader.initialize(new FileSplit(new ClassPathResource(dataFile).getFile()))
    reader
  }

  private def readData(recordReader: RecordReader) = Try {
    val labelIndex = 41
    val noOfClasses = 5
    val batchSize = 311010

    val iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, noOfClasses)
    val allData = iterator.next()
    allData.shuffle()
    allData
  }

  private def evaluate(model: MultiLayerNetwork, ds: DataSet): Try[Evaluation] = Try {
    val eval = new Evaluation(5)
    val output = model.output(ds.getFeatureMatrix())
    eval.eval(ds.getLabels(), output)
    eval
  }
}
