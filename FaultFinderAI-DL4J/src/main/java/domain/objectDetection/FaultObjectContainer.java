/**
 * 
 */
package domain.objectDetection;

import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.datavec.api.records.reader.RecordReader;
import org.deeplearning4j.nn.graph.ComputationGraph;

import clasDC.faults.FaultNames;
import clasDC.objects.CLASObject;
import clasDC.objects.CLASObject.ContainerType;
import clasDC.objects.SuperLayer;
import domain.models.CLASModelFactory;
import faultrecordreader.ImageSegmentationRecordReader;
import faultrecordreader.KunkelPetersFaultRecorderNew;
import faultrecordreader.MultiClassRecordReader;
import faultrecordreader.SlicingRecordReader;
import lombok.Builder;
import lombok.Getter;

/**
 * @author m.c.kunkel
 *
 * 
 *         FaultObjectContainer contains all the necessaries to run the
 *         model<br>
 *         input: CLASObject <br>
 *         output: RecordReader<br>
 *         output: ComputationGraph <br>
 */

public class FaultObjectContainer {
	@Getter
	private CLASObject clasObject = null;
	private CLASModelFactory factory = null;

	@Getter
	private ComputationGraph model = null;
	@Getter
	private int gridHeight;
	@Getter
	private int gridWidth;
	@Getter
	RecordReader recordReader = null;

	@Builder
	private FaultObjectContainer(CLASObject clasObject) {
		this.clasObject = clasObject;
		init();
	}

	private void init() {
		this.factory = CLASModelFactory.builder().clasObject(this.clasObject).build();
		this.model = factory.getComputationGraph();
		this.gridHeight = factory.getGridHeight();
		this.gridWidth = factory.getGridWidth();
		this.recordReader = setRecordReader();
	}

	public RecordReader setRecordReader() {
		if (clasObject.getContainerType().equals(ContainerType.CLASS)) {
			return new KunkelPetersFaultRecorderNew(this.clasObject);
		} else if (clasObject.getContainerType().equals(ContainerType.OBJ)) {
			return new SlicingRecordReader(this.clasObject);
		} else if (clasObject.getContainerType().equals(ContainerType.SEG)) {
			return new ImageSegmentationRecordReader(this.clasObject);
		} else if (clasObject.getContainerType().equals(ContainerType.MULTICLASS)) {
			return new MultiClassRecordReader(this.clasObject);
		} else {
			throw new IllegalArgumentException("The container type is not recognized " + clasObject.getContainerType());
		}
	}

	public static void main(String[] args) {
		CLASObject object = SuperLayer.builder().superlayer(1).nchannels(1).maxFaults(10)
				.desiredFaults(Stream.of(FaultNames.CONNECTOR_TREE, FaultNames.CONNECTOR_TREE, FaultNames.CHANNEL_THREE,
						FaultNames.PIN_SMALL).collect(Collectors.toCollection(ArrayList::new)))
				.singleFaultGen(false).containerType(ContainerType.SEG).build();
		FaultObjectContainer container = FaultObjectContainer.builder().clasObject(object).build();
	}

}
