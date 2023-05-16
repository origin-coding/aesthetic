import QtQuick.Controls.Material
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import QtQuick

ApplicationWindow {
    id: window
    visible: true; visibility:"Maximized"  // 默认最大化
    title: "图像美学质量评价系统 v1.0"

    Material.theme: Material.Dark

    menuBar: MenuBar{
        Menu{
            title: "帮助"
            // 关于选项被点击，打开关于对话框
            MenuItem{
                text: "关于"
                onTriggered: {
                    aboutDialog.open()
                }
            }
            // 源代码选项，点击后直接打开项目的GitHub仓库
            MenuItem{
                text: "源代码"
                onTriggered: {
                    Qt.openUrlExternally("https://github.com/QingyeKiyono/aesthetic")
                }
            }
        }
    }

    // 关于对话框，简单描述应用程序
    Dialog{
        id: aboutDialog
        anchors.centerIn: parent
        modal: true
        standardButtons: Dialog.Ok  // 只有确认选项

        title: "图像美学质量评价系统 v1.0，作者：孙强\n毕业论文《基于深度学习的图像美学质量评价系统设计与实现》界面部分"
    }

    // 文件选择框，选择需要进行评价的图像
    FileDialog {
        id: fileDialog
        title: "Select Image"
        nameFilters: [ "Images (*.png *.jpg *.bmp)" ]
        onAccepted: image.source = fileDialog.selectedFile  /*ImageDisplay.display_image(fileDialog.fileUrl)*/
    }

    RowLayout {
        Item {
            implicitHeight: window.height
            implicitWidth: window.width * 0.4

            // 展示图像
            Image {
                id: image
                width: window.width * 0.4
                height: window.height
                fillMode: Image.Stretch
                source: ""
            }
        }

        ColumnLayout{
            Button {
                text: "选择将要识别的图像"
                onClicked: {
                    fileDialog.open()
                }
            }
            Label {
                text: "选择的图片是：" + image.source
            }
            Label {
                text: "判断结果是："
            }
        }
    }
}
