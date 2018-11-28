package clasDC.factories;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.datavec.image.data.Image;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import clasDC.faults.Fault;
import clasDC.faults.FaultNames;
import clasDC.objects.CLASObject;
import clasDC.objects.DCSystem;
import clasDC.objects.DriftChamber;
import clasDC.objects.Region;
import clasDC.objects.SuperLayer;
import lombok.Builder;
import lombok.Getter;
import utils.FaultUtils;

@Getter
public class CLASFactoryImpl implements CLASFactory {

	private CLASObject clasObject = null;
	private CLASFactory clasFactory = null;
	private ComputationGraph model = null;

	@Builder
	public CLASFactoryImpl(CLASObject clasObject) {
		this.clasObject = clasObject;
		setObject();
	}

	public void setObject() {
		if (this.clasObject instanceof SuperLayer) {
			SuperLayer obj = (SuperLayer) this.clasObject;
			this.clasFactory = CLASSuperlayer.builder().superlayer(obj.getSuperlayer()).nchannels(obj.getNchannels())
					.maxFaults(obj.getMaxFaults()).desiredFaults(obj.getDesiredFaults())
					.singleFaultGen(obj.isSingleFaultGen()).build();
		} else if (this.clasObject instanceof DriftChamber) {
			DriftChamber obj = (DriftChamber) this.clasObject;
			this.clasFactory = CLASDriftChamber.builder().region(obj.getRegion()).nchannels(obj.getNchannels())
					.maxFaults(obj.getMaxFaults()).desiredFaults(obj.getDesiredFaults())
					.singleFaultGen(obj.isSingleFaultGen()).build();
		} else if (this.clasObject instanceof Region) {
			Region obj = (Region) this.clasObject;
			this.clasFactory = CLASDCRegion.builder().region(obj.getRegion()).nchannels(obj.getNchannels())
					.maxFaults(obj.getMaxFaults()).desiredFaults(obj.getDesiredFaults())
					.singleFaultGen(obj.isSingleFaultGen()).build();
		} else if (this.clasObject instanceof DCSystem) {
			DCSystem obj = (DCSystem) this.clasObject;
			this.clasFactory = CLASDCSystem.builder().nchannels(obj.getNchannels()).maxFaults(obj.getMaxFaults())
					.desiredFaults(obj.getDesiredFaults()).singleFaultGen(obj.isSingleFaultGen()).build();
		} else {
			throw new IllegalArgumentException("Invalid input: " + this.clasObject);
		}
	}

	public Image getImage() {
		return this.clasFactory.getImage();
	}

	public List<Fault> getFaultList() {
		return this.clasFactory.getFaultList();

	}

	public static void main(String[] args) {
		CLASObject object = SuperLayer.builder().superlayer(2).nchannels(1).maxFaults(10).singleFaultGen(true)
				.desiredFaults(Stream.of(FaultNames.CONNECTOR_TREE, FaultNames.CHANNEL_THREE, FaultNames.PIN_SMALL)
						.collect(Collectors.toCollection(ArrayList::new)))
				.build();

		CLASFactory factory = CLASFactoryImpl.builder().clasObject(object).build();
		// INDArray ret = factory.getObject().getImage().getImage();
		INDArray ret = factory.getImage().getImage();

		FaultUtils.draw(factory.getImage());
		int rank = ret.rank();
		int rows = (int) ret.size(rank == 3 ? 1 : 2);
		int cols = (int) ret.size(rank == 3 ? 2 : 3);
		int nchannels = (int) ret.size(rank == 3 ? 0 : 1);
		System.out.println(rank + " " + rows + " " + cols + " " + nchannels + "   " + ret.shapeInfoToString());

		for (Fault fault : factory.getFaultList()) {
			fault.printWireInformation();
		}
	}

}
