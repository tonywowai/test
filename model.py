import time
from typing import List, Dict, Optional
from aixblock_ml.model import AIxBlockMLBase
from ultralytics import YOLO
import os

css = """
@import "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css";

.aixblock__title .md p {
    display: block;
    text-align: center;
    font-size: 2em;
    font-weight: 700;
}

.aixblock__tabs .tab-nav {
    justify-content: center;
    gap: 8px;
    padding-bottom: 1rem;
    border-bottom: none !important;
}

.aixblock__tabs .tab-nav > button {
    border-radius: 8px;
    border: 1px solid #DEDEEC;
    height: 32px;
    padding: 8px 10px;
    text-align: center;
    line-height: 1em;
}

.aixblock__tabs .tab-nav > button.selected {
    background-color: #5050FF;
    border-color: #5050FF;
    color: #FFFFFF;
}

.aixblock__tabs .tabitem {
    padding: 0;
    border: none;
}

.aixblock__tabs .tabitem .gap.panel {
    background: none;
    padding: 0;
}

.aixblock__input-image,
.aixblock__output-image {
    border: solid 2px #DEDEEC !important;
}

.aixblock__input-image {
    border-style: dashed !important;
}

footer {
    display: none !important;
}

button.secondary,
button.primary {
    border: none !important;
    background-image: none !important;
    color: white !important;
    box-shadow: none !important;
    border-radius: 8px !important;
}

button.primary {
    background-color: #5050FF !important;
}

button.secondary {
    background-color: #F5A !important;
}

.aixblock__input-buttons {
    justify-content: flex-end;
}

.aixblock__input-buttons > button {
    flex: 0 !important;
}

.aixblock__trial-train {
    text-align: center;
    margin-top: 2rem;
}
"""

js = """
window.addEventListener("DOMContentLoaded", function() {
    function process() {
        let buttonsContainer = document.querySelector('.aixblock__input-image')?.parentElement?.nextElementSibling;
        
        if (!buttonsContainer) {
            setTimeout(function() {
                process();
            }, 100);
            return;
        }
        
        document.querySelectorAll('.aixblock__input-image').forEach(function(ele) {
            ele.parentElement.nextElementSibling.classList.add('aixblock__input-buttons');
        });
    }
    
    process();
});
"""


class MyModel(AIxBlockMLBase):

    is_init_model_trial = False
    model_trial_public_url = ""
    model_trial_local_url = ""

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ 
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}''')
        return []

    def fit(self, event, data, **kwargs):
        """

        
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

    def action(self, project, command, collection, **kwargs):

        print(f"""
              project: {project},
                command: {command},
                collection: {collection},
              """)
        if command.lower() == "train":
            try:
                checkpoint = kwargs.get("checkpoint")
                if checkpoint:
                    model = YOLO(f"checkpoints/uploads/{checkpoint}")
                else:
                    model = YOLO("yolov8n.pt")

                epochs = kwargs.get("epochs", 2)
                imgsz = kwargs.get("imgsz", 640)
                data = kwargs.get("data", "coco8.yaml")
                # check if folder named project exists
                if not os.path.exists(f"./yolov8/{project}"):
                    os.makedirs(f"./yolov8/{project}")
                result = model.train(data=data, imgsz=imgsz, epochs=epochs, project=f"./yolov8/{project}")
                return {"message": "train completed successfully"}
            except Exception as e:
                return {"message": f"train failed: {e}"}
        elif command.lower() == "predict":
            try:
                checkpoint = kwargs.get("checkpoint")
                if checkpoint:
                    model = YOLO(f"checkpoints/uploads/{checkpoint}")
                else:
                    model = YOLO("yolov8n.pt")

                data = kwargs.get("data", {})
                print(data)
                if data != {}:

                    img = data.get("img", "https://ultralytics.com/images/zidane.jpg")

                    result = model(img)

                    return {"message": "predict completed successfully",
                            "result": {"boxes": result[0].boxes.xyxy.tolist(),
                                       "names": result[0].names,
                                       "labels": result[0].boxes.cls.tolist()}}
                else:
                    return {"message": "predict failed", "result": None}
            except:
                return {"message": "predict failed", "result": None}
        else:
            return {"message": "command not supported", "result": None}

            # return {"message": "train completed successfully"}

    def model(self, project, **kwargs):
        import gradio as gr

        css = """
        .feedback .tab-nav {
            justify-content: center;
        }

        .feedback button.selected{
            background-color:rgb(115,0,254); !important;
            color: #ffff !important;
        }

        .feedback button{
            font-size: 16px !important;
            color: black !important;
            border-radius: 12px !important;
            display: block !important;
            margin-right: 17px !important;
            border: 1px solid var(--border-color-primary);
        }

        .feedback div {
            border: none !important;
            justify-content: center;
            margin-bottom: 5px;
        }

        .feedback .panel{
            background: none !important;
        }


        .feedback .unpadded_box{
            border-style: groove !important;
            width: 500px;
            height: 345px;
            margin: auto;
        }

        .feedback .secondary{
            background: rgb(225,0,170);
            color: #ffff !important;
        }

        .feedback .primary{
            background: rgb(115,0,254);
            color: #ffff !important;
        }

        .upload_image button{
            border: 1px var(--border-color-primary) !important;
        }
        .upload_image {
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }
        .upload_image .wrap{
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }

        .webcam_style .wrap{
            border: none !important;
            align-items: center !important;
            justify-content: center !important;
            height: 345px;
        }

        .webcam_style .feedback button{
            border: none !important;
            height: 345px;
        }

        .webcam_style .unpadded_box {
            all: unset !important;
        }

        .btn-custom {
            background: rgb(0,0,0) !important;
            color: #ffff !important;
            width: 200px;
        }

        .title1 {
            margin-right: 90px !important;
        }

        .title1 block{
            margin-right: 90px !important;
        }

        """

        with gr.Blocks(css=css) as demo2:
            with gr.Row():
                with gr.Column(scale=10):
                    gr.Markdown(
                        """
                        # Theme preview: `AIxBlock`
                        """
                    )

            import numpy as np

            def predict(input_img):
                import cv2
                result = self.action(project, "predict", collection="", data={"img": input_img})
                print(result)
                if result['result']:
                    boxes = result['result']['boxes']
                    names = result['result']['names']
                    labels = result['result']['labels']

                    for box, label in zip(boxes, labels):
                        box = [int(i) for i in box]
                        label = int(label)
                        input_img = cv2.rectangle(input_img, box, color=(255, 0, 0), thickness=2)
                        # input_img = cv2.(input_img, names[label], (box[0], box[1]), color=(255, 0, 0), size=1)
                        input_img = cv2.putText(input_img, names[label], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 255, 0), 2)

                return input_img

            def get_checkpoint_list(project):
                print("GETTING CHECKPOINT LIST")
                print(f"Proejct: {project}")
                import os
                checkpoint_list = [i for i in os.listdir("yolov8/models") if i.endswith(".pt")]
                checkpoint_list = [f"<a href='./yolov8/checkpoints/{i}' download>{i}</a>" for i in
                                   checkpoint_list]
                if os.path.exists(f"yolov8/{project}"):
                    for folder in os.listdir(f"yolov8/{project}"):
                        if "train" in folder:
                            project_checkpoint_list = [i for i in
                                                       os.listdir(f"yolov8/{project}/{folder}/weights") if
                                                       i.endswith(".pt")]
                            project_checkpoint_list = [
                                f"<a href='./yolov8/{project}/{folder}/weights/{i}' download>{folder}-{i}</a>"
                                for i in project_checkpoint_list]
                            checkpoint_list.extend(project_checkpoint_list)

                return "<br>".join(checkpoint_list)

            with gr.Tabs(elem_classes=["feedback"]) as parent_tabs:
                with gr.TabItem("Demo", id=0):
                    with gr.Row():
                        gr.Markdown("## Input", elem_classes=["title1"])
                        gr.Markdown("## Output", elem_classes=["title1"])

                    gr.Interface(predict,
                                 gr.Image(elem_classes=["upload_image"], sources="upload", container=False, height=345,
                                          show_label=False),
                                 gr.Image(elem_classes=["upload_image"], container=False, height=345, show_label=False),
                                 allow_flagging=False
                                 )

                with gr.TabItem("Webcam", id=1):
                    gr.Image(elem_classes=["webcam_style"], sources="webcam", container=False, show_label=False,
                             height=450)

                with gr.TabItem("Video", id=2):
                    gr.Image(elem_classes=["upload_image"], sources="clipboard", height=345, container=False,
                             show_label=False)

        gradio_app, local_url, share_url = demo2.launch(share=True, quiet=True, prevent_thread_lock=True,
                                                        server_name='0.0.0.0', show_error=True)

        return {"share_url": share_url, 'local_url': local_url}

    def model_trial(self, project, **kwargs):
        while self.is_init_model_trial:
            time.sleep(1)

        if len(self.model_trial_local_url) > 0 and len(self.model_trial_public_url) > 0:
            return {"share_url": self.model_trial_public_url, 'local_url': self.model_trial_local_url}

        self.is_init_model_trial = True
        print(kwargs)
        import gradio as gr

        def mt_predict(input_img):
            import cv2
            result = self.action(project, "predict", collection="", data={"img": input_img})
            print(result)
            if result['result']:
                boxes = result['result']['boxes']
                names = result['result']['names']
                labels = result['result']['labels']

                for box, label in zip(boxes, labels):
                    box = [int(i) for i in box]
                    label = int(label)
                    input_img = cv2.rectangle(input_img, box, color=(255, 0, 0), thickness=2)
                    # input_img = cv2.(input_img, names[label], (box[0], box[1]), color=(255, 0, 0), size=1)
                    input_img = cv2.putText(input_img, names[label], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)

            return input_img

        def mt_trial_training(dataset_choosen):
            print(f"Training with {dataset_choosen}")
            result = self.action(project, "train", collection="", data=dataset_choosen)
            return result['message']

        def mt_download_btn(evt: gr.SelectData):
            print(f"Downloading {dataset_choosen}")
            return f'<a href="/yolov8/datasets/{evt.value}" class="aixblock__download-button"><i class="fa fa-download"></i> Download</a>'

        def mt_get_checkpoint_list(project):
            print("GETTING CHECKPOINT LIST")
            print(f"Proejct: {project}")
            import os
            checkpoint_list = [i for i in os.listdir("yolov8/models") if i.endswith(".pt")]
            checkpoint_list = [f"<a href='./yolov8/checkpoints/{i}' download>{i}</a>" for i in
                               checkpoint_list]
            if os.path.exists(f"yolov8/{project}"):
                for folder in os.listdir(f"yolov8/{project}"):
                    if "train" in folder:
                        project_checkpoint_list = [i for i in
                                                   os.listdir(f"yolov8/{project}/{folder}/weights") if
                                                   i.endswith(".pt")]
                        project_checkpoint_list = [
                            f"<a href='./yolov8/{project}/{folder}/weights/{i}' download>{folder}-{i}</a>"
                            for i in project_checkpoint_list]
                        checkpoint_list.extend(project_checkpoint_list)

            return "<br>".join(checkpoint_list)

        def mt_tab_changed(tab):
            if tab == "Download":
                get_checkpoint_list(project=project)

        def mt_upload_file(file):
            return "File uploaded!"

        with gr.Blocks(css=css, js=js) as demo:
            gr.Markdown("AIxBlock", elem_classes=["aixblock__title"])

            with gr.Tabs(elem_classes=["aixblock__tabs"]):
                with gr.TabItem("Demo", id=0):
                    # with gr.Row():
                    #     gr.Markdown("Input", elem_classes=["aixblock__input-title"])
                    #     gr.Markdown("Output", elem_classes=["aixblock__output-title"])

                    gr.Interface(
                        mt_predict,
                        gr.Image(elem_classes=["aixblock__input-image"], container=False, height=345),
                        gr.Image(elem_classes=["aixblock__output-image"], container=False, height=345),
                        allow_flagging="never",
                    )

                # with gr.TabItem("Webcam", id=1):
                #     gr.Image(elem_classes=["webcam_style"], sources="webcam", container = False, show_label = False, height = 450)

                # with gr.TabItem("Video", id=2):
                #     gr.Image(elem_classes=["upload_image"], sources="clipboard", height = 345,container = False, show_label = False)

                # with gr.TabItem("About", id=3):
                #     gr.Label("About Page")

                with gr.TabItem("Trial Train", id=2):
                    gr.Markdown("Dataset template to prepare your own and initiate training")
                    # with gr.Row():
                    # get all filename in datasets folder
                    datasets = [(f"dataset{i}", name) for i, name in enumerate(os.listdir('./yolov8/datasets'))]
                    dataset_choosen = gr.Dropdown(datasets, label="Choose dataset", show_label=False, interactive=True,
                                                  type="value")
                    # gr.Button("Download this dataset", variant="primary").click(download_btn, dataset_choosen, gr.HTML())
                    download_link = gr.HTML("""<em>Please select a dataset to download</em>""")
                    dataset_choosen.select(mt_download_btn, None, download_link)

                    # when the button is clicked, download the dataset from dropdown
                    # download_btn
                    gr.Markdown("Upload your sample dataset to have a trial training")
                    # gr.File(file_types=['tar','zip'])

                    gr.Interface(
                        mt_predict,
                        gr.File(elem_classes=["aixblock__input-image"], file_types=['tar', 'zip'], container=False),
                        gr.Label(elem_classes=["aixblock__output-image"], container=False),
                        allow_flagging="never",
                    )

                    with gr.Column(elem_classes=["aixblock__trial-train"]):
                        gr.Button("Trial Train", variant="primary").click(mt_trial_training, dataset_choosen, None)
                        gr.HTML(value=f"<em>You can attemp up to {2} FLOps</em>")

                # with gr.TabItem("Download"):
                #     with gr.Column():
                #         gr.Markdown("## Download")
                #         with gr.Column():
                #             gr.HTML(get_checkpoint_list(project))

        gradio_app, local_url, share_url = demo.launch(share=True, quiet=True, prevent_thread_lock=True,
                                                       server_name='0.0.0.0', show_error=True)
        self.model_trial_public_url = share_url
        self.model_trial_local_url = local_url
        self.is_init_model_trial = False

        return {"share_url": share_url, 'local_url': local_url}

    def download(self, project, **kwargs):
        return super(self).download(project, **kwargs)
