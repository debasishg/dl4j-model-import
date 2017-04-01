lazy val commonSettings = Seq(
  version := "0.0.1",
  resolvers ++= Seq(
      Resolver.mavenLocal
    , Resolver.sonatypeRepo("releases")
    , Resolver.sonatypeRepo("snapshots")
    , "Bintray " at "https://dl.bintray.com/projectseptemberinc/maven"
  ),
  scalaVersion := "2.12.0",
  licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0")),
  libraryDependencies ++= Seq(
      "org.deeplearning4j"            % "deeplearning4j-core"           % "0.8.0",
      "org.deeplearning4j"            % "deeplearning4j-modelimport"    % "0.8.0",
      "org.nd4j"                      % "nd4j-native-platform"          % "0.8.0",
      "org.typelevel"                %% "cats"                          % "0.9.0",
      "org.slf4j"                     % "slf4j-log4j12"                 % "1.7.16"
    )
)

lazy val root = (project in file(".")).
  settings(commonSettings: _*).
  settings(
    name := "dl4j-model-import",
    scalacOptions ++= Seq(
      "-feature",
      "-unchecked",
      "-language:higherKinds",
      "-language:postfixOps",
      "-deprecation"
    )
  )

