"""
jwnb module and package to sdimulate wnb for the Jubaan back office: jwnb.jubaan.com
"""

# Python Example: Reading Entities
# Filterable fields: name, description, run_count, last_activity, notification_targets
from io import BytesIO
from pathlib import Path
import requests
from datetime import datetime, timezone
from typing import Any, Optional
from PIL import Image as PILImage
import numpy as np
import base64
import matplotlib.pyplot as plt

def make_api_request(entity, method='GET', data=None):
    url = f'https://app.base44.com/api/apps/6952664e26e9efd0556a45f7/entities/{entity}'
    headers = {
        'api_key': 'fd5723109ea1435d97553e56a7c953f9',
        'Content-Type': 'application/json'
    }
    if method.upper() == 'GET':
        response = requests.request(method, url, headers=headers, params=data)
    else:
        response = requests.request(method, url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()


def make_function_request(function_name, data=None):
    url = f'https://jwnb.base44.app/api/apps/6952664e26e9efd0556a45f7/functions/{function_name}'
    headers = {
        'api_key': 'fd5723109ea1435d97553e56a7c953f9',
        'Content-Type': 'application/json'
    }
    response = requests.request('POST', url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()



# Python Example: Updating an Entity
# Filterable fields: name, description, run_count, last_activity, notification_targets
def update_entity(entity, entity_id, update_data):
    response = requests.put(
        f'https://app.base44.com/api/apps/6952664e26e9efd0556a45f7/entities/{entity}/{entity_id}',
        headers={
            'api_key': 'fd5723109ea1435d97553e56a7c953f9',
            'Content-Type': 'application/json'
        },
        json=update_data
    )
    response.raise_for_status()
    return response.json()


class jwnb:

    def __init__(self, project_name, description="", run_name="Run", tags=[], config={}, notes="", total_epochs=10):
        self.project_name = project_name
        self.description = description
        self.project = self.__get_or_create_project__()    

        # init the new run
        run_name = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.run = self.__create_run__(run_name, notes, config=config, tags=tags, total_epochs=total_epochs)

#region Project Management

    def __get_or_create_project__(self):
        pdata = {"name": self.project_name}
        projects = make_api_request(f'Project', data=pdata)
        if projects:
            return projects[0]
        else:
            pdata = {"name": self.project_name, "description": self.description}
            project = make_api_request(f'Project', method="POST", data=pdata)
            return project

    def update_project_notification(self, notification_targets):
        update_data = {"notification_targets": notification_targets}
        updated_project = update_entity('Project', self.project['id'], update_data)
        self.project = updated_project
        return updated_project
#endregion

#region Run Management

    def __get_system_info__(self) -> dict:
        """Gather system information."""
        import platform       
        
        info = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
        }
        
        # Try to get GPU info
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
                info["cuda"] = torch.version.cuda
                info["pytorch"] = torch.__version__
        except ImportError:
            pass
            
        try:
            import tensorflow as tf
            info["tensorflow"] = tf.__version__
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                info["gpu_count"] = len(gpus)
        except ImportError:
            pass
            
        return info

    def __create_run__(self, run_name, notes="", status="running", config={}, tags=[], total_epochs=10, summary={}):
        
        system_info = self.__get_system_info__()
        stnow = datetime.now(timezone.utc).isoformat()
        run_data = {
            "run_name": run_name,
            "project_name": self.project['name'],
            "notes": notes,            
            "tags": tags,
            "status": status,
            "config": config,
            "start_time": stnow,
            "current_epoch": 0,
            "total_epochs": total_epochs,
            "system_info": system_info,
            "summary": summary
        }
        run = make_api_request('Run', method="POST", data=run_data)
        return run

    def update_current_epoch(self, current_epoch):
        update_data = {"current_epoch": current_epoch}
        updated_run = update_entity('Run', self.run['id'], update_data)
        self.run = updated_run
        return updated_run

    def finish(self, status="finished", summary={}, error:str=""):
        etnow = datetime.now(timezone.utc).isoformat()
        update_data = {
            "status": status,
            "end_time": etnow,
            "summary": summary,
            "notes": self.run['notes'] + (f"<br><hr><br><b>Error</b><font color=\"red\">: {error}</font>" if error else "")
        }
        updated_run = update_entity('Run', self.run['id'], update_data)
        self.run = updated_run

        # inform notifications
        make_function_request('notifyRunStatus', data={"run_name": self.run['run_name']})
        return updated_run

#endregion

#region Metrics

    class Image:
        """
        Log images from matplotlib figures, PIL images, or numpy arrays.
        Similar to wandb.Image()
        
        Args:
            data: matplotlib figure, PIL Image, numpy array, or file path
            caption: Optional caption for the image
            
        Example:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(losses)
            jwnb.log(value_type="image", value=jwnb.Image(fig), epoch=epoch)
        """
        def __init__(self, data: Any, scale=1.0):
            self.data = self.__to_base64__(data)
            return 

        def __to_base64__(self, data:Any) -> str:
            """Convert image data to base64 string for upload."""
            buf = BytesIO()
            
            # Handle matplotlib figure
            if hasattr(data, 'savefig'):
                data.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
            
            # Handle PIL Image
            elif PILImage is not None and isinstance(data, PILImage.Image):
                data.save(buf, format='PNG')
                buf.seek(0)
            
            # Handle numpy array
            elif np is not None and hasattr(data, '__array__'):
                arr = np.asarray(data)
                if arr.ndim == 2:  # Grayscale
                    arr = np.stack([arr] * 3, axis=-1)
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                if PILImage is not None:
                    img = PILImage.fromarray(arr.astype(np.uint8))
                    img.save(buf, format='PNG')
                    buf.seek(0)
                else:
                    raise ImportError("PIL is required for numpy array images")
            
            # Handle file path
            elif isinstance(data, (str, Path)):
                with open(data, 'rb') as f:
                    return base64.b64encode(f.read()).decode()
            else:
                raise ValueError(f"Unsupported image type: {type(data)}")

            return base64.b64encode(buf.read()).decode()

    # scalar, histogram, image, model, list, string, boolean
    def log(self, value_type, value:Any, caption:str="", step: int = None):

        log_data = {
            "run_name": self.run['run_name'],
            "value_type": value_type,
            "value": str(value),
            "value_caption": caption,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # define epoch or take from current
        if step is not None:
            log_data["epoch"] = step
            if step != self.run['current_epoch']:
                self.update_current_epoch(step)
        else:
            log_data["epoch"] = self.run['current_epoch']

        log_entry = make_api_request('Epoch', method="POST", data=log_data)
        return log_entry

#endregion

if __name__ == "__main__":
    
    config = {"learning_rate": 0.001, "batch_size": 32, "optimizer": "adam"}
    numepochs = 5
    jwnb_instance = jwnb("Dudi Sample Project", "This is a sample project description.", "dudis-run", tags=["dudi", "sample", "test"], total_epochs=numepochs, config=config, notes="This is a sample run for testing purposes.")
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

            jwnb_instance.log(value_type="scalar", value=loss, caption="loss", step=epc)   
            jwnb_instance.log(value_type="scalar", value=acc, caption="accuracy", step=epc)   
            
            list_data = [np.random.rand() for _ in range(10)]
            jwnb_instance.log(value_type="list", value=list_data, caption="list of values", step=epc)

            # loass and acc data          
            histdata = {"bins": bins, "counts": [np.random.randint(0, 100) for _ in range(len(bins)-1)]}
            jwnb_instance.log(value_type="histogram", value=histdata, caption="rand hist", step=epc)

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
            jwnb_instance.log(value_type="image", value=jwnb.Image(fig).data, caption="Loss and Accuracy Plots", step=epc)
            plt.close(fig)

            # changes in the model during epoch run
            model = {"layers": [{"type": "Dense", "units": 64}, {"type": "Dense", "units": 10}]}
            jwnb_instance.log(value_type="model", value=model, caption="hyper params",step=epc)
            jwnb_instance.log(value_type="string", value=f"this is a string for epoch {epc}", step=epc)
        
        # end of all epochs, update loss and acc (or other params)
        summary = {"accuracy": 0.92, "loss": 0.15}
        jwnb_instance.finish(status="finished", summary=summary)


    except Exception as e:
        jwnb_instance.finish(status="crashed", error=str(e))
        print(f"Run failed: {e}")