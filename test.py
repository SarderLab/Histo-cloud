import os

def del_all_xml_files(gc, folder_id):
    items = list(gc.listItem(folder_id))
    
    for item in items:
        slide_name = item['name'].split(".")[0]
        files = list(gc.listFile(item['_id']))
        for file in files:
            if file['name'] == "{}.xml".format(slide_name):
                gc.delete("/file/{}".format(file['_id']))
                print("Deleted {}".format(file['name']))

def get_girder_instance(url, token):
    import girder_client
    gc = girder_client.GirderClient(apiUrl=url)
    gc.setToken(token)
    return gc

def del_xmls_and_pngs_from_svs(path):
    for folder in os.listdir(path):
            if folder.endswith(".svs"):
                files = os.listdir(os.path.join(path, folder))
                for file in files:
                    if file.endswith(".xml") or file.endswith(".png"):
                        os.remove(os.path.join(path, folder, file))
                        print("Deleted {}".format(file))
                

def main(url, token):
    # gc = get_girder_instance(url, token)
    # del_all_xml_files(gc, "67900d7958b173229fd609cc")

    path = "/orange/pinaki.sarder/anish.tatke/Histo-cloudTN/LNR01_Test"
    del_xmls_and_pngs_from_svs(path)


    
        


            

if __name__ == "__main__":
    GIRDER_URL = "https://devathena.rc.ufl.edu/api/v1"
    GIRDER_TOKEN = "0EzHs4NNN66IOK84Otbgr3bhTpQ4D5jTgO0JJKXOAPpJ7xXDonNcvef9JHG0qjsw"

    main(GIRDER_URL, GIRDER_TOKEN)