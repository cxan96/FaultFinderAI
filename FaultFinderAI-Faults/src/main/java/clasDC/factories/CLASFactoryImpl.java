package clasDC.factories;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.datavec.image.data.Image;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import clasDC.faults.Fault;
import clasDC.faults.FaultNames;
import clasDC.objects.CLASObject;
import clasDC.objects.CLASObject.ContainerType;
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
					.minFaults(obj.getMinFaults()).maxFaults(obj.getMaxFaults()).desiredFault(obj.getDesiredFault())
					.desiredFaults(obj.getDesiredFaults()).singleFaultGen(obj.isSingleFaultGen())
					.isScaled(obj.isScaled()).desiredFaultGenRate(obj.getDesiredFaultGenRate()).build();
		} else if (this.clasObject instanceof DriftChamber) {
			DriftChamber obj = (DriftChamber) this.clasObject;
			this.clasFactory = CLASDriftChamber.builder().region(obj.getRegion()).nchannels(obj.getNchannels())
					.minFaults(obj.getMinFaults()).maxFaults(obj.getMaxFaults()).desiredFaults(obj.getDesiredFaults())
					.singleFaultGen(obj.isSingleFaultGen()).build();
		} else if (this.clasObject instanceof Region) {
			Region obj = (Region) this.clasObject;
			this.clasFactory = CLASDCRegion.builder().region(obj.getRegion()).nchannels(obj.getNchannels())
					.minFaults(obj.getMinFaults()).maxFaults(obj.getMaxFaults()).desiredFaults(obj.getDesiredFaults())
					.singleFaultGen(obj.isSingleFaultGen()).build();
		} else if (this.clasObject instanceof DCSystem) {
			DCSystem obj = (DCSystem) this.clasObject;
			this.clasFactory = CLASDCSystem.builder().nchannels(obj.getNchannels()).minFaults(obj.getMinFaults())
					.maxFaults(obj.getMaxFaults()).desiredFaults(obj.getDesiredFaults())
					.singleFaultGen(obj.isSingleFaultGen()).build();
		} else {
			throw new IllegalArgumentException("Invalid input: " + this.clasObject + "  for a clasObject");
		}
	}

	public Image getImage() {
		return this.clasFactory.getImage();
	}

	public List<Fault> getFaultList() {
		return this.clasFactory.getFaultList();

	}

	public INDArray getSegmentationLabels() {
		return this.clasFactory.getSegmentationLabels();
	}

	@Override
	public Map<FaultNames, INDArray> faultLocationLabels() {
		return this.clasFactory.faultLocationLabels();

	}

	@Override
	public Map<String, INDArray> locationLabels() {
		return this.clasFactory.locationLabels();

	}

	public CLASFactory getNewFactory() {
		this.clasFactory = this.clasFactory.getNewFactory();
		return this.clasFactory;

	}

	public static void main(String[] args) {
		FaultNames desiredFault = FaultNames.CHANNEL_THREE;
		CLASObject object = SuperLayer.builder().superlayer(3).nchannels(1).minFaults(2).maxFaults(2)
				.desiredFaults(Stream.of(FaultNames.FUSE_A, FaultNames.FUSE_B, FaultNames.FUSE_C,
						FaultNames.CONNECTOR_TREE, FaultNames.CONNECTOR_THREE, FaultNames.CONNECTOR_E,
						FaultNames.CHANNEL_ONE, FaultNames.CHANNEL_TWO, FaultNames.CHANNEL_THREE, FaultNames.PIN_BIG,
						FaultNames.PIN_SMALL).collect(Collectors.toCollection(ArrayList::new)))
				.singleFaultGen(false).containerType(ContainerType.SEG).desiredFault(desiredFault)
				.desiredFaultGenRate(0.45).build();

		CLASFactory factory = CLASFactoryImpl.builder().clasObject(object).build();
		for (int i = 0; i < 10; i++) {
			FaultUtils.draw(factory.getImage());
			System.out.println(factory.faultLocationLabels().get(desiredFault));
			factory = factory.getNewFactory();
		}

	}

}
