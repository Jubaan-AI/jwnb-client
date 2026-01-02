
from jwnb import jwnb
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    config = {"learning_rate": 0.001, "batch_size": 32, "optimizer": "adam"}
    numepochs = 5
    jwnb_instance = jwnb("Jubaan ML Gixam", "This is a sample project description.", "jwnb-run", tags=["jubaan", "sample", "test"], total_epochs=numepochs, config=config, notes="This is a sample run for testing purposes.")
    jwnb_instance.update_project_notification(["david@jubaan.com"])


    bins = [i for i in range(0, 101, 10)]    
    
    losscount = []
    acccount = []

    try:
        for epc in range(1, numepochs + 1):
        
            # sumuilating an epoch run
            loss = np.random.rand() + (numepochs - epc) * 0.1
            acc = np.random.rand() * 0.1 + (epc / numepochs) * 0.9

            # loging
            losscount.append(loss)  
            acccount.append(acc)

            # sample for scalar values
            jwnb_instance.log(value_type="scalar", value=loss, caption="loss", step=epc)   
            jwnb_instance.log(value_type="scalar", value=acc, caption="accuracy", step=epc)  
            jwnb_instance.log(value_type="scalar", value=np.random.rand() * 0.1 + (epc / numepochs) * 0.6, caption="stupidity", step=epc)   

            # sample for boolean values
            jwnb_instance.log(value_type="boolean", value=bool(acc > 0.5), caption="fuzzy boolean", step=epc)  

            # sample for list of values            
            list_data = [np.random.rand() for _ in range(10)]
            jwnb_instance.log(value_type="list", value=list_data, caption="list of values", step=epc)

            # loss and acc data - sample of histogram
            histdata = {"bins": bins, "counts": [np.random.randint(0, 100) for _ in range(len(bins)-1)]}
            jwnb_instance.log(value_type="histogram", value=histdata, caption="rand hist", step=epc)

            # sample of charts
            # {'xaxis': {'data': [1, 2, 3, 4], 'title': 'x-axis'}, 'yaxis': {['data': [69, 86, 66, 56]}, 'legend': 'Random Chart'], "title":"y-axis"}
            chartdata = {"xaxis": 
                            {"data":[i for i in range(1, epc, 1)], "title": "x-axis"},  
                         "yaxis": 
                            [
                                {"data": [np.random.randint(0, 100) for _ in range(epc - 1)], "legend": "Random values 1"}, 
                                {"data": [np.random.randint(0, 100) for _ in range(epc - 1)], "legend": "Random values 2"}
                            ], "title": "y-axis"}
            jwnb_instance.log(value_type="chart", value=chartdata, caption="Random Chart", step=epc)

            # create plots from loass and acc
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(range(len(losscount)), losscount, label='Loss', color='blue')
            ax[0].set_title('Loss over Epochs') 
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Loss')    
            ax[0].legend()
            ax[1].plot(range(len(acccount)), acccount, label='Accuracy', color='green')
            ax[1].set_title('Accuracy over Epochs')         
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('Accuracy')    
            ax[1].legend()

            # sample of image type
            jwnb_instance.log(value_type="image", value=jwnb_instance.Image(fig), caption="Loss and Accuracy Plots", step=epc)
            plt.close(fig)

            # sample of model (any json serializable object) and string
            model = {"layers": [{"type": "Dense", "units": 64}, {"type": "Dense", "units": 10}]}
            jwnb_instance.log(value_type="model", value=model, caption="hyper params",step=epc)
            jwnb_instance.log(value_type="string", value=f"this is a string for epoch {epc}", step=epc)
            print(f"Completed epoch {epc}/{numepochs}")
        
        # end of all epochs, update loss and acc (or other params)
        summary = {"accuracy": acc, "loss": loss, "stupidity": 0.67}
        jwnb_instance.finish(status="finished", summary=summary)


    except Exception as e:
        jwnb_instance.finish(status="crashed", error=str(e))
        print(f"Run failed: {e}")