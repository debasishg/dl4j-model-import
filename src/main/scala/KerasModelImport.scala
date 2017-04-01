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

import cats._
import cats.data._
import cats.implicits._

object KerasModelImportExample {
  case class DataAttributes(
    numLinesToSkipInFile: Int,
    delimiter: String,
    labelIndex: Int,
    noOfClasses: Int,
    batchSize: Int
  )

  def main(args: Array[String]): Unit = {
    val modelJsonPath = "/Users/debasishghosh/projects/dl4j-model-import/src/main/resources/model.json"
    val weightsH5Path = "/Users/debasishghosh/projects/dl4j-model-import/src/main/resources/model.h5"
    val dataFile = "kdd.csv"
    val attributes = DataAttributes(0, ",", 41, 5, 311010)

    val stats = for {
      model  <- importFromKeras(modelJsonPath, weightsH5Path)
      reader <- getReader(dataFile)
      data   <- readData(reader)
      eval   <- evaluate(model, data)
    } yield eval.stats()

    (Try { attributes } >>= stats.run) foreach println
  }

  private def importFromKeras(jsonPath: String, h5Path: String)
    : Kleisli[Try, DataAttributes, MultiLayerNetwork] = Kleisli { (da: DataAttributes) =>
    Try {
      KerasModelImport.importKerasSequentialModelAndWeights(jsonPath, h5Path, false)
    }
  }

  private def getReader(dataFile: String)
    : Kleisli[Try, DataAttributes, RecordReader] = Kleisli { (da: DataAttributes) =>
    Try {
      val reader = new CSVRecordReader(da.numLinesToSkipInFile, da.delimiter)
      reader.initialize(new FileSplit(new ClassPathResource(dataFile).getFile()))
      reader
    }
  }

  private def readData(recordReader: RecordReader)
    : Kleisli[Try, DataAttributes, DataSet] = Kleisli { (da: DataAttributes) =>
    Try {
      val iterator = new RecordReaderDataSetIterator(recordReader, da.batchSize, da.labelIndex, da.noOfClasses)
      val allData = iterator.next()
      allData.shuffle()
      allData
    }
  }

  private def evaluate(model: MultiLayerNetwork, ds: DataSet)
    : Kleisli[Try, DataAttributes, Evaluation] = Kleisli { (da: DataAttributes) =>
    Try {
      val eval = new Evaluation(da.noOfClasses)
      val output = model.output(ds.getFeatureMatrix())
      eval.eval(ds.getLabels(), output)
      eval
    }
  }
}
