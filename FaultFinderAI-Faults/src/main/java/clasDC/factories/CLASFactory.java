package clasDC.factories;

import java.util.List;
import java.util.Map;

import org.datavec.image.data.Image;
import org.nd4j.linalg.api.ndarray.INDArray;

import clasDC.faults.Fault;
import clasDC.faults.FaultNames;

public interface CLASFactory {

	public Image getImage();

	public List<Fault> getFaultList();

	public INDArray getSegmentationLabels();

	public Map<FaultNames, INDArray> faultLocationLabels();

	public Map<String, INDArray> locationLabels();

	public CLASFactory getNewFactory();

}
