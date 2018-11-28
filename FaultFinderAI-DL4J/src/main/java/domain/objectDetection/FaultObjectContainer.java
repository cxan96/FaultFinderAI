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
import clasDC.objects.DriftChamber;
import domain.models.CLASModelFactory;
import faultrecordreader.CLASObjectRecordReader;
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

	// CLASObjectRecordReader
	@Builder
	private FaultObjectContainer(CLASObject clasObject) {
		this.clasObject = clasObject;
		init();
	}

	private void init() {
		this.factory = CLASModelFactory.builder().clasOject(this.clasObject).build();
		this.model = factory.getComputationGraph();
		this.gridHeight = factory.getGridHeight();
		this.gridWidth = factory.getGridWidth();
		this.recordReader = new CLASObjectRecordReader(clasObject, this.gridHeight, this.gridWidth);

	}

	public static void main(String[] args) {
		CLASObject object = DriftChamber.builder().region(1).nchannels(1).maxFaults(10)
				.desiredFaults(Stream.of(FaultNames.CONNECTOR_TREE, FaultNames.CONNECTOR_TREE, FaultNames.CHANNEL_THREE,
						FaultNames.PIN_SMALL).collect(Collectors.toCollection(ArrayList::new)))
				.singleFaultGen(false).build();
		FaultObjectContainer container = FaultObjectContainer.builder().clasObject(object).build();
		System.out.println(container.getGridHeight() + "   " + container.getGridWidth());
	}

}
